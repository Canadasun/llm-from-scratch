#!/usr/bin/env bash
#
# Canary-deploy a new checkpoint to the live Azure Container Apps endpoint and
# (optionally) promote it with a gradual traffic cutover. This encodes the PROVEN
# 2026-10-31 path used to ship the instruction-tuned M3 candidate, so a re-run
# can't re-stumble on the hazards we actually hit.
#
# Two phases, each a separate, explicit invocation (promotion is never automatic):
#
#   # 1) build + push + deploy a 0-traffic canary on an ISOLATED key store, verify
#   CKPT=checkpoints/instruction-candidate/ckpt.pt TAG=m3-instruct-canary \
#     ./deploy/canary.sh canary
#
#   # 2) promote it: gradual 90/10 -> 100, then deactivate the old revisions
#   TAG=m3-instruct-canary ./deploy/canary.sh promote
#
# WHY A CANARY NEEDS ITS OWN KEY DB (single-writer invariant):
#   The persistent key store is SQLite over Azure Files (SMB) opened with
#   ?nolock=1 (SMB/CIFS can't take SQLite's byte-range locks) and pinned to
#   max-replicas=1 -> exactly ONE writer. A canary that mounts the SAME keys.db
#   while the live revision is also up = two writers with locking DISABLED =
#   corruption. So in phase (1) the canary points at its OWN keys-canary.db.
#
# WHY A GRADUAL (both-serving) PROMOTION IS NEVERTHELESS SAFE on the shared db:
#   The /generate hot-path is READ-ONLY on the key store -- store.authenticate()
#   is a pure SELECT (enterprise/sqlstore.py) and the audit sink is stdout JSON
#   lines, NOT the DB (serve/__main__.py). Concurrent *reads* over SMB are fine;
#   only *writes* (key issuance/rotation, or the first-run demo-key mint on an
#   EMPTY store) need the single writer. On a NON-EMPTY persistent store startup
#   does NOT re-mint (serve/__main__.py) so existing keys stay valid across the
#   cutover. => phase (2) repoints the canary at the shared keys.db and splits
#   traffic; DO NOT issue/rotate keys during the split window.
#   At the end it DEACTIVATES the old revisions so only one can hold keys.db.
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
SUFFIX="${SUFFIX:-candidate}"            # promoted revision suffix (shared store)
CANARY_SUFFIX="${CANARY_SUFFIX:-instruct-canary}"  # 0-traffic verify revision
# Serving env the live endpoint runs with (match GA): secured + quantized + metrics.
SHARED_DB="file:/data/keys.db?nolock=1"
CANARY_DB="file:/data/keys-canary.db?nolock=1"

az account set --subscription "$SUBSCRIPTION"

build_and_push() {
  [ -f "$CKPT" ] || { echo "ERROR: $CKPT not found"; exit 1; }
  # Bake the candidate as checkpoints/ckpt.pt WITHOUT disturbing the on-disk GA
  # checkpoint or the committed .dockerignore; restore both on any exit. Exclude
  # the candidate working dir so a multi-hundred-MB last.pt can't bloat the image.
  local bak di_bak
  bak="$(mktemp)"; di_bak="$(mktemp)"
  cp checkpoints/ckpt.pt "$bak"; cp .dockerignore "$di_bak"
  trap 'cp "$bak" checkpoints/ckpt.pt; cp "$di_bak" .dockerignore; rm -f "$bak" "$di_bak"' RETURN
  printf '\n# (canary build, temporary)\ncheckpoints/instruction-candidate/\n' >> .dockerignore
  cp "$CKPT" checkpoints/ckpt.pt
  echo ">> baking $(sha256sum checkpoints/ckpt.pt | cut -c1-12) into $IMAGE"
  docker build -t "$IMAGE" .
  docker push "$IMAGE"   # package inherits existing PUBLIC visibility of :<app>
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

case "$PHASE" in
  canary)
    build_and_push
    echo ">> deploying 0-traffic canary revision --$CANARY_SUFFIX on ISOLATED $CANARY_DB"
    az containerapp update -n "$APP" -g "$RG" --image "$IMAGE" --revision-suffix "$CANARY_SUFFIX" \
      --set-env-vars SECURED=1 "KEY_DB=$CANARY_DB" QUANTIZE=1 METRICS=1 -o none
    # Multiple-revision mode + pinned traffic => the new revision gets 0 traffic.
    rev="$APP--$CANARY_SUFFIX"; f="$(fqdn_of "$rev")"
    wait_healthz "$f"
    echo ">> canary up at https://$f (0 traffic). Verify auth (no-key->401, key->200) then:"
    echo "   TAG=$TAG ./deploy/canary.sh promote"
    ;;
  promote)
    echo ">> creating promotion revision --$SUFFIX on SHARED $SHARED_DB (no re-mint on non-empty store)"
    az containerapp update -n "$APP" -g "$RG" --image "$IMAGE" --revision-suffix "$SUFFIX" \
      --set-env-vars SECURED=1 "KEY_DB=$SHARED_DB" QUANTIZE=1 METRICS=1 -o none
    new="$APP--$SUFFIX"; wait_healthz "$(fqdn_of "$new")"
    # Current live (100% traffic) revision = the one to step down from.
    old="$(az containerapp ingress traffic show -n "$APP" -g "$RG" \
           --query "[?weight==\`100\`].revisionName | [0]" -o tsv)"
    [ -n "$old" ] && [ "$old" != "$new" ] || { echo "ERROR: could not find a single 100% revision to step down"; exit 1; }
    echo ">> gradual cutover: $old=90 / $new=10  (observe, then full)"
    az containerapp ingress traffic set -n "$APP" -g "$RG" --revision-weight "$old=90" "$new=10" -o none
    echo "   (send test traffic through the MAIN fqdn now; expect 200s, zero 401s)"
    sleep 15
    echo ">> full cutover: $new=100 / $old=0"
    az containerapp ingress traffic set -n "$APP" -g "$RG" --revision-weight "$new=100" "$old=0" -o none
    # Restore strict single-writer: deactivate every OTHER active revision so only
    # the promoted one can ever hold keys.db open.
    for r in $(az containerapp revision list -n "$APP" -g "$RG" \
                 --query "[?properties.active && name!='$new'].name" -o tsv); do
      az containerapp revision deactivate -n "$APP" -g "$RG" --revision "$r" -o none && echo "   deactivated $r"
    done
    url="$(az containerapp show -n "$APP" -g "$RG" --query properties.configuration.ingress.fqdn -o tsv)"
    echo ">> PROMOTED. Live: https://$url  (only $new active)"
    echo ">> if the canary used an isolated keys-canary.db, delete it (az resolves the key internally):"
    echo "   az storage file delete --share-name keys --account-name <acct> --path keys-canary.db"
    ;;
  *)
    echo "usage: $0 {canary|promote}   (see header for env vars)"; exit 2;;
esac
