#!/usr/bin/env bash
# Local build / release qualification entry point. Azure budget: $0.
# canary/promote are intentionally blocked before any cloud command. The rollout
# implementation below is retained for the Azure owner's future review; it is
# NOT operationally qualified and must not be enabled under the current budget.
#
# CKPT=checkpoints/ckpt.pt IMAGE=llm-local:candidate ./deploy/canary.sh build
# IMAGE=repo@sha256:... RELEASE_RECORD=release.json ./deploy/canary.sh qualify
# IMAGE=repo@sha256:... RELEASE_RECORD=release.json ./deploy/canary.sh verify
set -euo pipefail

PHASE="${1:-canary}"

SUBSCRIPTION="${SUBSCRIPTION:-202808c8-8440-4d93-b646-3ed956877a85}"
RG="${RG:-llm-from-scratch-rg}"
APP="${APP:-llm-serve}"
GH_USER="${GH_USER:-canadasun}"
TAG="${TAG:-m3-instruct-canary}"
IMAGE="${IMAGE:-ghcr.io/${GH_USER}/${APP}:${TAG}}"
CKPT="${CKPT:-checkpoints/instruction-candidate/ckpt.pt}"
SUFFIX="${SUFFIX:-candidate}"
CANARY_SUFFIX="${CANARY_SUFFIX:-instruct-canary}"
SHARED_DB="${SHARED_DB:-}"
CANARY_DB="${CANARY_DB:-}"
# Promotion gate thresholds (measured on the new revision before full cutover).
GATE_SAMPLES="${GATE_SAMPLES:-20}"
MIN_SUCCESS_RATE="${MIN_SUCCESS_RATE:-0.95}"
MAX_P95_MS="${MAX_P95_MS:-3000}"
PROBE_KEY="${PROBE_KEY:-}"   # mandatory for any future promotion

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
# This repository is operated under a $0 Azure budget. Offline work remains
# available; cloud rollouts cannot guarantee zero incremental cost and are blocked.
case "$PHASE" in
  build|qualify|verify) ;;
  canary|promote)
    echo 'BLOCKED: Azure budget is $0; no cloud rollout or overlapping revisions permitted.'
    echo "Use local build/qualify/verify. Azure owner must resolve storage, cost and operations separately."
    exit 1 ;;
  *) echo "usage: $0 {build|qualify|verify|canary|promote}"; exit 2 ;;
esac
if [ "$PHASE" = qualify ] || [ "$PHASE" = verify ]; then
  : "${RELEASE_RECORD:?RELEASE_RECORD is required}"
  exec python -m deploy.release "$PHASE" --image "$IMAGE" --record "$RELEASE_RECORD"
fi

build_local() {
  [ -f "$CKPT" ] || { echo "ERROR: $CKPT not found"; exit 1; }
  command -v rsync >/dev/null || { echo "ERROR: rsync is required for an isolated build context"; exit 1; }
  BUILD_CONTEXT="$(mktemp -d)"
  local ctx="$BUILD_CONTEXT"
  # EXIT (not RETURN) so cleanup runs even when set -e aborts on a failed build.
  trap 'rm -rf "$BUILD_CONTEXT"' EXIT
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
  echo ">> Built locally. No push or Azure operation performed."
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
  python -m deploy.probe --base "https://$1" --samples "$GATE_SAMPLES" \
    --min-success "$MIN_SUCCESS_RATE" --max-p95-ms "$MAX_P95_MS"

}

case "$PHASE" in
  build)
    build_local
    ;;
  canary)
    : "${RELEASE_RECORD:?passing release evidence is required}"
    python -m deploy.release verify --image "$IMAGE" --record "$RELEASE_RECORD"
    : "${CANARY_DB:?locking-capable isolated storage is required}"
    [[ "$CANARY_DB" != *nolock* ]] || exit 1
    az account set --subscription "$SUBSCRIPTION"
    az containerapp revision set-mode -n "$APP" -g "$RG" --mode multiple -o none
    echo ">> deploying 0-traffic canary --$CANARY_SUFFIX on ISOLATED $CANARY_DB"
    az containerapp update -n "$APP" -g "$RG" --image "$IMAGE" --revision-suffix "$CANARY_SUFFIX" \
      --set-env-vars SECURED=1 "KEY_DB=$CANARY_DB" QUANTIZE=1 METRICS=1 -o none
    rev="$APP--$CANARY_SUFFIX"
    wait_healthz "$(fqdn_of "$rev")"
    echo ">> canary up (0 traffic). Verify, then: TAG=$TAG ./deploy/canary.sh promote"
    ;;
  promote)
    : "${RELEASE_RECORD:?passing release evidence is required}"
    python -m deploy.release verify --image "$IMAGE" --record "$RELEASE_RECORD"
    : "${SHARED_DB:?supported storage is required}"
    : "${PROBE_KEY:?authenticated quality probes are mandatory}"
    [[ "$SHARED_DB" != *nolock* ]] || exit 1
    az account set --subscription "$SUBSCRIPTION"
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
