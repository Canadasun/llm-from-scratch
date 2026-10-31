# Serving (Milestone 4)

An inference endpoint for the from-scratch model. The design keeps all the
decision logic in a **framework-agnostic, unit-tested core** and uses the HTTP
layer as a thin shell, so the hard parts (validation, rate limiting, safety,
generation) are testable without a running server and the transport can be
swapped later without touching them.

```
request JSON
   │
   ▼
validation.py   reject wrong types; clamp ranges (prompt ≤ 8k chars,
   │            max_new_tokens ≤ 512, temperature ∈ [0.05, 2.0], top_k, seed)
   ▼
ratelimit.py    per-client token bucket (burst + sustained rate)
   │
   ▼
content_policy  use-safety on the PROMPT  → block (403) / flag (caution) / allow
   │
   ▼
engine.py       secure-load model (weights_only=True); generate continuation
   │
   ▼
content_policy  use-safety on the OUTPUT  → withhold (403) / caution / allow
   │
   ▼
response JSON   completion, token counts, params, request_id, latency_ms, caution?
```

## Run it

```bash
python -m serve                       # 127.0.0.1:8000, checkpoints/ckpt.pt
python -m serve --port 8080 --ckpt path/to/ckpt.pt
python -m serve --device cuda         # if you have a GPU
python -m serve --host 0.0.0.0        # expose — be deliberate about this
python -m serve --no-safety           # research mode; logged on startup
```

## API

### `GET /healthz`
```json
{"status":"ok","block_size":256,"vocab_size":1024,"tokenizer":"bpe","safety":true}
```

### `POST /generate`
Request (all fields except `prompt` optional; shown with defaults):
```json
{"prompt":"Zero trust architecture ",
 "max_new_tokens":128,"temperature":0.8,"top_k":null,"seed":null}
```
```bash
curl -s localhost:8000/generate -H 'content-type: application/json' \
     -d '{"prompt":"Zero trust architecture ","max_new_tokens":80,"seed":1}'
```
Success (`200`):
```json
{"completion":"...","prompt_tokens":12,"completion_tokens":80,
 "params":{"max_new_tokens":80,"temperature":0.8,"top_k":null,"seed":1},
 "request_id":"3aed5e4e6fe4","latency_ms":1893.1}
```
A flagged (dual-use) prompt still returns `200` but adds a `"caution"` field.

### Error responses
| status | `error` | when |
|---|---|---|
| 400 | `invalid_request` | bad JSON, wrong type, out-of-range field, unknown field |
| 403 | `blocked_prompt` / `withheld_output` | use-safety policy refused |
| 413 | `payload_too_large` | request body over 64 KB |
| 429 | `rate_limited` | client exceeded its token bucket |
| 500 | `generation_failed` | internal error (details are logged, not returned) |

Every request produces one structured JSON log line (`rid`, `client`, `status`,
`ms`, and `error`/`completion_tokens`).

## Security posture
- **Untrusted input:** every field is validated and hard-bounded before use.
- **Use-safety on both sides:** prompt and generated output are checked
  (`safety/content_policy.py`), dual-use-aware (security topics are allowed;
  clearly malicious content is blocked).
- **The model only returns text** — it never executes anything.
- **Secure checkpoint load:** `weights_only=True` (no pickle code execution).
- **No `X-Forwarded-For` trust:** the client key is the socket peer, so the rate
  limiter can't be bypassed with a spoofed header. Put this behind a trusted
  proxy if you need real client IPs.
- **Bind to `127.0.0.1` by default;** exposing it is an explicit `--host` choice.
- **Optional authentication (`--secured`):** wraps the service in the enterprise
  security layer (`serve/secured.py`) so `/generate` requires an
  `Authorization: Bearer <key>`, authorizes the `model:infer` capability by role
  + key scope, rate-limits per authenticated principal (not socket IP), and
  audits every request. Without the flag the endpoint is open (local-dev
  default). See [`ENTERPRISE_SECURITY.md`](ENTERPRISE_SECURITY.md).

## Honest limitations
- This stdlib server is **threaded, single-process** — fine for a demo or a local
  dev endpoint. For production the same `InferenceService` drops behind
  FastAPI/uvicorn or gunicorn with multiple workers; **only `http_server.py` is
  replaced**. That swap, plus auth, TLS termination at a proxy, and a shared
  (e.g. Redis) rate-limit store, are the remaining production items.
- The rate limiter is in-process, so it does not coordinate across workers.
- Model **capability** is unchanged by serving it — see `MODEL_CARD.md`.
