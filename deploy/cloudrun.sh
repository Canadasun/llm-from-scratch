#!/usr/bin/env bash
#
# Deploy the from-scratch LLM inference endpoint (serve/) to Google Cloud Run.
#
# Design: CPU-only, scale-to-zero (min-instances 0) so it costs ~$0 when idle
# and stays within the free tier for light use; capped at a couple of instances
# so it can never run away on cost.
#
# Security posture: the service is deployed REQUIRING authentication (default
# Cloud Run IAM) — it is NOT public. Invoke it with an identity token, or grant
# a specific principal the run.invoker role. Making it world-reachable is a
# deliberate, separate decision (see the note at the bottom); this script will
# not do it for you.
#
# Prerequisites (done by YOU, once):
#   - a billing-enabled GCP project (the free $300 trial counts)
#   - gcloud CLI authenticated:  gcloud auth login
#   - checkpoints/ckpt.pt present locally (the M3 model, baked into the image)
#
# Usage (nothing hard-coded to one account):
#   PROJECT=<your-project-id> REGION=us-central1 ./deploy/cloudrun.sh
#
set -euo pipefail

PROJECT="${PROJECT:?set PROJECT=<your-gcp-project-id>}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-llm-serve}"

echo ">> project=$PROJECT region=$REGION service=$SERVICE"

gcloud config set project "$PROJECT"

# APIs needed to build from source and run a container.
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com

# Build from the local source (uploads the dir minus .gcloudignore, which keeps
# checkpoints/ckpt.pt), then deploy. Authenticated by default (no public access).
gcloud run deploy "$SERVICE" \
  --source . \
  --region "$REGION" \
  --no-allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --min-instances 0 \
  --max-instances 2 \
  --timeout 120 \
  --port 8080

echo ">> done. The service requires auth. Test it with your own identity token:"
echo "   URL=\$(gcloud run services describe $SERVICE --region $REGION --format='value(status.url)')"
echo "   TOKEN=\$(gcloud auth print-identity-token)"
echo "   curl -H \"Authorization: Bearer \$TOKEN\" \$URL/healthz"
echo
echo ">> To expose it publicly LATER (deliberate choice), you would run:"
echo "   gcloud run services add-iam-policy-binding $SERVICE --region $REGION \\"
echo "     --member=allUsers --role=roles/run.invoker"
