#!/usr/bin/env bash
#
# Deploy the from-scratch LLM inference endpoint (serve/) to Azure Container Apps.
#
# This is the PROVEN path (2026-10-31 deploy). Every step below encodes a fix for
# an error we actually hit, so a re-run can't re-stumble on them:
#   - deploy from a PUBLIC registry image (--image), NEVER `--source` — `--source`
#     provisions a Basic ACR (~$5/mo standing cost) that blows the $5 cap.
#   - `az containerapp up` does NOT accept --only-show-errors (fails arg parse).
#   - scale-to-zero (min-replicas 0) + a hard max keep idle cost at ~$0.
#   - the image must be built, pushed to GHCR, and made PUBLIC first (see
#     build_and_push.sh); GHCR push needs a classic PAT with write:packages.
#   - a $5 budget with alerts is created so cost can never surprise you (budgets
#     ALERT, they do not hard-stop — the real cap is the architecture above).
#   - SECURED=1 runs the enterprise auth layer so the public URL needs an API key.
#
# Usage:
#   SUBSCRIPTION=<personal-sub-id> ./deploy/containerapp.sh
#
set -euo pipefail

# The user requires a $0 Azure budget. This provisioning script can create
# billable storage/revisions; budgets do not hard-stop charges. Refuse before
# parameter expansion, credential reads, az calls, or resource mutation.
echo 'BLOCKED: Azure budget is $0. Provisioning/deployment is disabled.' >&2
echo 'Use local qualification; retain live changes in the Azure owner workstream.' >&2
exit 1

SUBSCRIPTION="${SUBSCRIPTION:?set SUBSCRIPTION=<personal-azure-sub-id>}"
RG="${RG:-llm-from-scratch-rg}"
LOCATION="${LOCATION:-canadacentral}"
ENVIRONMENT="${ENVIRONMENT:-llm-env}"
APP="${APP:-llm-serve}"
IMAGE="${IMAGE:-ghcr.io/canadasun/llm-serve:m3-persist2}"  # PUBLIC; has the persistent key store
BUDGET_CAD="${BUDGET_CAD:-5}"
ALERT_EMAIL="${ALERT_EMAIL:-idwu1470@gmail.com}"
# Persistent key store: an Azure Files share holds the SQLite key DB so API keys
# survive restarts/recycles (without this, --secured mints a DIFFERENT demo key
# per replica/restart -> intermittent 401s and key loss). The account name is
# DETERMINISTIC (derived from sub+RG) so re-running this script REUSES the same
# storage + key DB instead of creating a fresh, empty one (which would silently
# drop every existing credential). Override STORAGE_ACCOUNT to pin a specific name.
STORAGE_ACCOUNT="${STORAGE_ACCOUNT:-stllmkeys$(printf '%s' "$SUBSCRIPTION$RG" | sha1sum | cut -c1-12)}"
FILE_SHARE="${FILE_SHARE:-keys}"
STORAGE_MOUNT="${STORAGE_MOUNT:-keysvol}"
# The key store now REFUSES nolock (SQLite's own locking must be available). We pin
# to ONE writer (max-replicas=1) and the app enforces a single-writer lease. NOTE:
# plain SQLite over Azure Files (SMB) may not provide byte-range locks at all; if the
# store fails to open on the mount, move persistence to local-disk (single replica)
# or a managed transactional DB (Postgres). Cloud rollout is blocked under the $0
# Azure budget regardless (see docs/AUDIT_REMEDIATION_2026-11-01.md).
KEY_DB_URI="${KEY_DB_URI:-file:/data/keys.db}"

echo ">> subscription=$SUBSCRIPTION rg=$RG app=$APP image=$IMAGE"
az account set --subscription "$SUBSCRIPTION"

az group create -n "$RG" -l "$LOCATION" \
  --tags Environment=Personal Project=llm-from-scratch -o none
echo ">> RG ready"

# Create env + app pulling the PUBLIC image. NOTE: no --only-show-errors here
# (unsupported by `up`); no --source (would create a paid ACR).
az containerapp up \
  --name "$APP" \
  --resource-group "$RG" \
  --location "$LOCATION" \
  --environment "$ENVIRONMENT" \
  --image "$IMAGE" \
  --target-port 8080 \
  --ingress external \
  --env-vars SECURED=1

# --- Persistent key store on Azure Files (tiny, ~$0; needs Microsoft.Storage) ---
az provider register --namespace Microsoft.Storage -o none 2>/dev/null || true
az storage account create -n "$STORAGE_ACCOUNT" -g "$RG" -l "$LOCATION" \
  --sku Standard_LRS --kind StorageV2 --min-tls-version TLS1_2 \
  --allow-blob-public-access false -o none
SA_KEY="$(az storage account keys list --account-name "$STORAGE_ACCOUNT" -g "$RG" --query '[0].value' -o tsv)"
az storage share create -n "$FILE_SHARE" --account-name "$STORAGE_ACCOUNT" --account-key "$SA_KEY" --quota 1 -o none
az containerapp env storage set -n "$ENVIRONMENT" -g "$RG" \
  --storage-name "$STORAGE_MOUNT" --azure-file-account-name "$STORAGE_ACCOUNT" \
  --azure-file-account-key "$SA_KEY" --azure-file-share-name "$FILE_SHARE" \
  --access-mode ReadWrite -o none
echo ">> Azure Files share '$FILE_SHARE' mounted as env storage '$STORAGE_MOUNT'"

# Apply image + mount + KEY_DB + scale together via YAML, so the new revision
# comes up with the mount already present (KEY_DB at /data must exist at start).
# SINGLE writer: max-replicas=1 (SQLite-over-SMB is not safe for concurrent
# writers). scale-to-zero (min 0) keeps idle cost ~$0; 2Gi so torch won't OOM.
TMP_YAML="$(mktemp)"
az containerapp show -n "$APP" -g "$RG" -o yaml > "$TMP_YAML"
python3 - "$TMP_YAML" "$IMAGE" "$KEY_DB_URI" "$STORAGE_MOUNT" <<'PY'
import sys, yaml
f, image, keydb, mount = sys.argv[1:5]
d = yaml.safe_load(open(f)); t = d["properties"]["template"]; c = t["containers"][0]
c["image"] = image
c["env"] = [{"name": "SECURED", "value": "1"}, {"name": "KEY_DB", "value": keydb}]
c["volumeMounts"] = [{"volumeName": mount, "mountPath": "/data"}]
t["volumes"] = [{"name": mount, "storageType": "AzureFile", "storageName": mount}]
t["scale"]["minReplicas"] = 0; t["scale"]["maxReplicas"] = 1
for k in ["provisioningState","runningStatus","latestRevisionName","latestReadyRevisionName",
          "latestRevisionFqdn","customDomainVerificationId","eventStreamEndpoint",
          "outboundIpAddresses","delegatedIdentities","patchingMode"]:
    d["properties"].pop(k, None)
d.pop("systemData", None)
yaml.safe_dump(d, open(f, "w"), sort_keys=False)
PY
az containerapp update -n "$APP" -g "$RG" --yaml "$TMP_YAML" -o none
rm -f "$TMP_YAML"
echo ">> persistence live: image=$IMAGE KEY_DB=$KEY_DB_URI scale min=0 max=1 cpu=1 mem=2Gi"

# $5 budget + actual-cost alerts (50/80/100%). Uses create-with-rg + --time-period;
# NOTE the schema has no thresholdType in this CLI, so we use actual-cost alerts.
START="$(date -u +%Y-%m-01T00:00:00Z)"
END="$(date -u -d '+2 years' +%Y-%m-01T00:00:00Z 2>/dev/null || date -u -v+2y +%Y-%m-01T00:00:00Z)"
az consumption budget create-with-rg \
  -g "$RG" -n "llm-monthly-${BUDGET_CAD}cad" \
  --amount "$BUDGET_CAD" --category Cost --time-grain Monthly \
  --time-period start-date="$START" end-date="$END" \
  --notifications "{\"Actual_50\":{\"enabled\":true,\"operator\":\"GreaterThan\",\"threshold\":50,\"contactEmails\":[\"$ALERT_EMAIL\"]},\"Actual_80\":{\"enabled\":true,\"operator\":\"GreaterThan\",\"threshold\":80,\"contactEmails\":[\"$ALERT_EMAIL\"]},\"Actual_100\":{\"enabled\":true,\"operator\":\"GreaterThan\",\"threshold\":100,\"contactEmails\":[\"$ALERT_EMAIL\"]}}" \
  -o none || echo "   (budget may already exist; not fatal)"
echo ">> \$$BUDGET_CAD budget + alerts set"

URL="$(az containerapp show -n "$APP" -g "$RG" --query properties.configuration.ingress.fqdn -o tsv)"
echo ">> LIVE: https://$URL"
echo ">> demo API key (SECURED): az containerapp logs show -n $APP -g $RG --tail 60 | grep -o 'sk_[A-Za-z0-9_-]*'"
echo ">> test: curl https://$URL/healthz"
