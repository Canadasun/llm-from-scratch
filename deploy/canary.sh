#!/usr/bin/env bash
#
# Canary-deploy a new checkpoint to the live Azure Container Apps endpoint and
# (optionally) promote it with a gradual, GATED traffic cutover and automatic
# rollback. Two explicit phases (promotion is never automatic):
#
#   # 1) build + push + deploy a 0-traffic canary on an ISOLATED key store, verify
#   CKPT=checkpoints/instruction-candidate/ckpt.pt TAG=m3-instruct-canary \
#     ./deploy/canary.sh canary
#
#   # 2) promote it: 90/10 -> measured gate -> 100, else automatic rollback
#   TAG=m3-instruct-canary ./deploy/canary.sh promote
#
# DESIGN NOTES THAT ENCODE PRIOR AUDIT FINDINGS:
#
# * Isolated build context (audit "deployment integrity"): the image is built
#   from a TEMP directory containing the repo plus the candidate baked in as
#   checkpoints/ckpt.pt. The working tree's real checkpoint and .dockerignore are
#   NEVER modified, so there is nothing to restore and no cleanup trap on which
#   correctness depends. The only trap removes the temp dir and uses EXIT (which
#   DOES fire when `set -e` aborts after a failed build/push), unlike a RETURN
#   trap. Even if that cleanup is skipped, only a temp dir leaks.
#
# * Gated promotion with rollback (audit "release safety"): before sending full
#   traffic, the promotion revision is measured on its OWN direct FQDN -- the
#   EXACT deployed image/model -- for health, enforced auth (401 without a key),
#   error rate and p95 latency (and, if PROBE_KEY is supplied, a real authed
#   generation). Failing any gate triggers an automatic rollback: traffic returns
#   to the previous revision and the new one is deactivated.
#
# * Single-writer key store (audit "persistence"): the /generate hot-path is
#   read-only on the key store, so a 90/10 split may share keys.db SAFELY as long
#   as no key issuance/rotation runs in the window; the app also enforces a writer
#   lease (enterprise/writer_lease.py) so two writers cannot both initialise/write.
#   The canary verify phase still uses an isolated keys-canary.db.
#
set -euo pipefail

PHASE="${1:-canary}"

SUBSCRIPTION="${SUBSCRIPTION:-202808c8-8440-4d93-b646-3ed956877a85}"
RG="${RG:-llm-from-scratch-rg}"
APP="${APP:-llm-serve}"
GH_USER="${GH_USER:-canadasun}"
TAG="${TAG:-m3-instruct-canary}"
IMAGE="ghcr.io/${GH_USER}/${APP}:${TAG}"
CKPT="${CKPT:-checkpoints/instruction-candidate/ckpt.pt}"
SUFFIX="${SUFFIX:-candidate}"
CANARY_SUFFIX="${CANARY_SUFFIX:-instruct-canary}"
SHARED_DB="file:/data/keys.db?nolock=1"
CANARY_DB="file:/data/keys-canary.db?nolock=1"
# Promotion gate thresholds (measured on the new revision before full cutover).
GATE_SAMPLES="${GATE_SAMPLES:-20}"
MIN_SUCCESS_RATE="${MIN_SUCCESS_RATE:-0.95}"
MAX_P95_MS="${MAX_P95_MS:-3000}"
PROBE_KEY="${PROBE_KEY:-}"   # optional: enables a real authed /generate check

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
az account set --subscription "$SUBSCRIPTION"

build_and_push() {
  [ -f "$CKPT" ] || { echo "ERROR: $CKPT not found"; exit 1; }
  command -v rsync >/dev/null || { echo "ERROR: rsync is required for an isolated build context"; exit 1; }
  local ctx
  ctx="$(mktemp -d)"
  # EXIT (not RETURN) so cleanup runs even when set -e aborts on a failed build.
  trap 'rm -rf "$ctx"' EXIT
  # Copy only what the image needs; exclude heavy/dev trees and ALL checkpoints,
  # then overlay ONLY the candidate as the baked checkpoint. The working tree is
  # untouched.
  rsync -a \
    --exclude '.git' --exclude '.venv' --exclude 'checkpoints' \
    --exclude 'data' --exclude 'notebooks' --exclude 'docs' --exclude 'tests' \
    --exclude '__pycache__' --exclude '*.pyc' --exclude '*.pt' \
    "$REPO_ROOT"/ "$ctx"/
  mkdir -p "$ctx/checkpoints"
  cp "$CKPT" "$ctx/checkpoints/ckpt.pt"
  echo ">> building $IMAGE from isolated context (candidate sha $(sha256sum "$CKPT" | cut -c1-12))"
  docker build -t "$IMAGE" "$ctx"
  docker push "$IMAGE"   # tag inherits the package's existing PUBLIC visibility
}

fqdn_of() { az containerapp revision show -n "$APP" -g "$RG" --revision "$1" --query properties.fqdn -o tsv; }

wait_healthz() {
  local url="https://$1/healthz" i code
  for i in $(seq 1 18); do
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 "$url" 2>/dev/null || echo 000)
    echo "   healthz $1: $code (try $i)"; [ "$code" = 200 ] && return 0; sleep 5
  done
  echo "ERROR: $1 never became healthy"; return 1
}

# Measure the EXACT deployed revision on its own FQDN. Returns 0 if every gate
# passes, 1 otherwise. No secrets handled unless PROBE_KEY is explicitly provided.
gate_revision() {
  local fqdn="$1" base="https://$1" ok=0 n="$GATE_SAMPLES" code lat sum=0 i p95
  echo ">> gating revision at $base ($n samples; min_success=$MIN_SUCCESS_RATE p95<=${MAX_P95_MS}ms)"
  # Enforced auth: a request WITHOUT a key must be rejected (401).
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 -X POST "$base/generate" \
         -H 'content-type: application/json' -d '{"prompt":"x","max_new_tokens":4}' || echo 000)
  [ "$code" = 401 ] || { echo "   FAIL: unauthenticated /generate returned $code, expected 401"; return 1; }
  echo "   auth gate OK (no-key -> 401)"
  # Availability + latency over N health probes (no secret needed).
  local lats=()
  for i in $(seq 1 "$n"); do
    out=$(curl -s -o /dev/null -w '%{http_code} %{time_total}' --max-time 20 "$base/healthz" 2>/dev/null || echo "000 20")
    code=${out%% *}; lat=${out##* }
    if [ "$code" = 200 ]; then ok=$((ok+1)); lats+=("$lat"); fi
  done
  local rate; rate=$(awk "BEGIN{print $ok/$n}")
  # p95 of successful latencies (seconds -> ms)
  p95=$(printf '%s\n' "${lats[@]:-0}" | sort -n | awk '{a[NR]=$1} END{if(NR==0){print 99999;exit} i=int(0.95*NR); if(i<1)i=1; print a[i]*1000}')
  echo "   success_rate=$rate  p95=${p95}ms  ($ok/$n)"
  awk "BEGIN{exit !($rate >= $MIN_SUCCESS_RATE)}" || { echo "   FAIL: success rate below $MIN_SUCCESS_RATE"; return 1; }
  awk "BEGIN{exit !($p95 <= $MAX_P95_MS)}" || { echo "   FAIL: p95 latency above ${MAX_P95_MS}ms"; return 1; }
  # Optional deep check: a real authed generation (only if the operator supplies a key).
  if [ -n "$PROBE_KEY" ]; then
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 40 -X POST "$base/generate" \
           -H 'content-type: application/json' -H "authorization: Bearer $PROBE_KEY" \
           -d '{"prompt":"Q: What is TLS?\nA: ","max_new_tokens":8}' || echo 000)
    [ "$code" = 200 ] || { echo "   FAIL: authed /generate returned $code, expected 200"; return 1; }
    echo "   functional gate OK (authed /generate -> 200)"
  else
    echo "   (set PROBE_KEY to also gate a real authed generation)"
  fi
  echo ">> all promotion gates PASSED"; return 0
}

case "$PHASE" in
  canary)
    build_and_push
    echo ">> deploying 0-traffic canary --$CANARY_SUFFIX on ISOLATED $CANARY_DB"
    az containerapp update -n "$APP" -g "$RG" --image "$IMAGE" --revision-suffix "$CANARY_SUFFIX" \
      --set-env-vars SECURED=1 "KEY_DB=$CANARY_DB" QUANTIZE=1 METRICS=1 -o none
    rev="$APP--$CANARY_SUFFIX"
    wait_healthz "$(fqdn_of "$rev")"
    echo ">> canary up (0 traffic). Verify, then: TAG=$TAG ./deploy/canary.sh promote"
    ;;
  promote)
    echo ">> creating promotion revision --$SUFFIX on SHARED $SHARED_DB"
    az containerapp update -n "$APP" -g "$RG" --image "$IMAGE" --revision-suffix "$SUFFIX" \
      --set-env-vars SECURED=1 "KEY_DB=$SHARED_DB" QUANTIZE=1 METRICS=1 -o none
    new="$APP--$SUFFIX"; new_fqdn="$(fqdn_of "$new")"
    wait_healthz "$new_fqdn"
    old="$(az containerapp ingress traffic show -n "$APP" -g "$RG" \
           --query "[?weight==\`100\`].revisionName | [0]" -o tsv)"
    [ -n "$old" ] && [ "$old" != "$new" ] || { echo "ERROR: no single 100% revision to step down from"; exit 1; }

    echo ">> 90/10 canary split: $old=90 / $new=10"
    az containerapp ingress traffic set -n "$APP" -g "$RG" --revision-weight "$old=90" "$new=10" -o none

    # GATE the exact deployed revision. On failure, ROLL BACK automatically.
    if ! gate_revision "$new_fqdn"; then
      echo ">> ROLLBACK: restoring $old=100, $new=0, deactivating $new"
      az containerapp ingress traffic set -n "$APP" -g "$RG" --revision-weight "$old=100" "$new=0" -o none
      az containerapp revision deactivate -n "$APP" -g "$RG" --revision "$new" -o none || true
      echo ">> promotion ABORTED; live traffic unchanged on $old"
      exit 1
    fi

    echo ">> gates passed; full cutover: $new=100 / $old=0"
    az containerapp ingress traffic set -n "$APP" -g "$RG" --revision-weight "$new=100" "$old=0" -o none
    for r in $(az containerapp revision list -n "$APP" -g "$RG" \
                 --query "[?properties.active && name!='$new'].name" -o tsv); do
      az containerapp revision deactivate -n "$APP" -g "$RG" --revision "$r" -o none && echo "   deactivated $r"
    done
    url="$(az containerapp show -n "$APP" -g "$RG" --query properties.configuration.ingress.fqdn -o tsv)"
    echo ">> PROMOTED. Live: https://$url  (only $new active)"
    ;;
  *)
    echo "usage: $0 {canary|promote}   (see header for env vars)"; exit 2;;
esac
