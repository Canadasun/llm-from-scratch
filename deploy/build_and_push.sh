#!/usr/bin/env bash
#
# Build the serving image (checkpoint baked in) and push it to GHCR, then you make
# it PUBLIC so Azure Container Apps can pull it with no credentials (no paid ACR).
#
# Encodes the fixes for errors we hit:
#   - GHCR push needs a token with write:packages. The `gh` OAuth token does NOT
#     have it by default and `gh auth refresh -s write:packages` may not stick
#     (GH_TOKEN env can override). USE A CLASSIC PAT with write:packages+read:packages.
#   - PYTHONUNBUFFERED=1 + print(flush=True) (already in Dockerfile / __main__) so
#     the --secured demo key actually shows up in container logs.
#   - .dockerignore keeps checkpoints/ckpt.pt but drops .venv/.git/data to keep the
#     image lean.
#
# Usage:
#   GHCR_PAT=<classic-PAT-with-write:packages> ./deploy/build_and_push.sh
#   # then make the package public (Danger Zone) at:
#   #   https://github.com/users/Canadasun/packages/container/llm-serve/settings
#
set -euo pipefail

IMAGE="${IMAGE:-ghcr.io/canadasun/llm-serve:m3}"
GH_USER="${GH_USER:-Canadasun}"
CKPT="${CKPT:-checkpoints/ckpt.pt}"

[ -f "$CKPT" ] || { echo "ERROR: $CKPT not found (the trained checkpoint must exist locally)"; exit 1; }

echo ">> building $IMAGE (this bakes in $CKPT)"
docker build -t "$IMAGE" .

echo ">> logging in to GHCR"
if [ -n "${GHCR_PAT:-}" ]; then
  echo "$GHCR_PAT" | docker login ghcr.io -u "$GH_USER" --password-stdin
else
  echo "   GHCR_PAT not set — run this yourself (keeps the token out of any agent):"
  echo "     echo <classic-PAT-with-write:packages> | docker login ghcr.io -u $GH_USER --password-stdin"
  echo "   then re-run with the login already done."
fi

echo ">> pushing $IMAGE"
docker push "$IMAGE"

echo ">> DONE. Now make the package PUBLIC (one-time), then run deploy/containerapp.sh:"
echo "   https://github.com/users/$GH_USER/packages/container/llm-serve/settings  (Danger Zone -> Change visibility -> Public)"
