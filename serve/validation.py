"""
Validate and hard-bound an untrusted /generate request body.

Every field that arrives over the wire is untrusted. This module rejects wrong
types and clamps/limits ranges BEFORE any value reaches the model, so a client
cannot, for example, ask for a billion tokens, a zero/negative temperature, or
megabytes of prompt. It returns a clean params dict or raises ValidationError.
"""

import math

# Hard limits. These are ceilings, not suggestions.
MAX_PROMPT_CHARS = 8000
MAX_NEW_TOKENS = 512
MAX_TEMPERATURE = 2.0
MIN_TEMPERATURE = 0.05

DEFAULTS = {
    "max_new_tokens": 128,
    "temperature": 0.8,
    "top_k": None,
    "seed": None,
}

_ALLOWED_KEYS = {"prompt", "max_new_tokens", "temperature", "top_k", "seed"}


class ValidationError(ValueError):
    """Raised for a malformed or out-of-bounds request. Message is client-safe."""


def _as_int(name, value, lo, hi, allow_none=False):
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{name} must be an integer")
    if value < lo or value > hi:
        raise ValidationError(f"{name} must be between {lo} and {hi}")
    return value


def validate_generate(payload):
    """payload: the parsed JSON object from the request body. Returns a dict with
    keys: prompt, max_new_tokens, temperature, top_k, seed."""
    if not isinstance(payload, dict):
        raise ValidationError("request body must be a JSON object")

    unknown = set(payload) - _ALLOWED_KEYS
    if unknown:
        raise ValidationError(f"unknown field(s): {', '.join(sorted(unknown))}")

    prompt = payload.get("prompt", "")
    if not isinstance(prompt, str):
        raise ValidationError("prompt must be a string")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise ValidationError(
            f"prompt too long ({len(prompt)} chars; max {MAX_PROMPT_CHARS})")

    max_new_tokens = _as_int(
        "max_new_tokens", payload.get("max_new_tokens", DEFAULTS["max_new_tokens"]),
        1, MAX_NEW_TOKENS)

    temperature = payload.get("temperature", DEFAULTS["temperature"])
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ValidationError("temperature must be a number")
    try:
        # A huge JSON integer (e.g. 10**400) overflows float() with OverflowError;
        # turn that into a clean validation error instead of a 500.
        temperature = float(temperature)
    except (OverflowError, ValueError):
        raise ValidationError("temperature is out of range")
    if math.isnan(temperature) or math.isinf(temperature):
        raise ValidationError("temperature must be a finite number")
    if temperature < MIN_TEMPERATURE or temperature > MAX_TEMPERATURE:
        raise ValidationError(
            f"temperature must be between {MIN_TEMPERATURE} and {MAX_TEMPERATURE}")

    top_k = _as_int("top_k", payload.get("top_k", DEFAULTS["top_k"]),
                    1, 1_000_000, allow_none=True)

    seed = _as_int("seed", payload.get("seed", DEFAULTS["seed"]),
                   0, 2**31 - 1, allow_none=True)

    return {
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "seed": seed,
    }
