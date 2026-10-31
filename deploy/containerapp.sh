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

SUBSCRIPTION="${SUBSCRIPTION:?set SUBSCRIPTION=<personal-azure-sub-id>}"
RG="${RG:-llm-from-scratch-rg}"
LOCATION="${LOCATION:-canadacentral}"
ENVIRONMENT="${ENVIRONMENT:-llm-env}"
APP="${APP:-llm-serve}"
IMAGE="${IMAGE:-ghcr.io/canadasun/llm-serve:m3}"   # must be PUBLIC (no ACR cost)
BUDGET_CAD="${BUDGET_CAD:-5}"
ALERT_EMAIL="${ALERT_EMAIL:-idwu1470@gmail.com}"

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

# Enforce scale-to-zero, a replica cap, and enough memory that torch won't OOM
# on cold start.
az containerapp update -n "$APP" -g "$RG" \
  --min-replicas 0 --max-replicas 2 --cpu 1.0 --memory 2.0Gi -o none
echo ">> scale set: min=0 max=2 cpu=1 mem=2Gi"

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
