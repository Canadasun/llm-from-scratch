# Cloud Run (or any container host) image for the from-scratch LLM inference
# endpoint. CPU-only: the model is small and serving needs just PyTorch plus the
# repo's own hand-written modules. The trained checkpoint is baked in at
# checkpoints/ckpt.pt (see .gcloudignore, which keeps that one file).
FROM python:3.13-slim

WORKDIR /app
# Needed for independent evaluation of this same immutable serving image.
RUN apt-get update && apt-get install -y --no-install-recommends bubblewrap libseccomp2 \
    && rm -rf /var/lib/apt/lists/*

# CPU build of PyTorch only (no CUDA, no pymupdf — that is a corpus-build dep,
# not a serving dep). Pinned to the CPU wheel index.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# App code + the baked-in checkpoint (filtered by .gcloudignore / .dockerignore).
COPY . /app

# Cloud Run injects $PORT (defaults to 8080). Bind all interfaces INSIDE the
# container — the only network path in is Cloud Run's managed proxy.
ENV PORT=8080
# Unbuffered stdout/stderr so logs (incl. the --secured demo key line) appear in
# the container log immediately rather than being held in a buffer.
ENV PYTHONUNBUFFERED=1

# exec form via sh so ${PORT} expands at runtime. Setting SECURED=1 turns on the
# enterprise auth layer (API-key required, tenant-scoped, audited) — used when
# the endpoint is reachable from the internet.
# SECURED=1 turns on API-key auth. KEY_DB=/path turns on the PERSISTENT SQLite key
# store (keys survive restarts); for it to persist across replica recycles on a
# cloud host, point KEY_DB at a MOUNTED volume (e.g. Azure Files), not the
# container's ephemeral local fs.
CMD ["sh", "-c", "exec python -m serve --host 0.0.0.0 --port ${PORT} --ckpt checkpoints/ckpt.pt ${SECURED:+--secured} ${KEY_DB:+--key-db ${KEY_DB}} ${METRICS:+--metrics} ${QUANTIZE:+--quantize}"]
