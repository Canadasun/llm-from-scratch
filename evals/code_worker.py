"""Sandbox-only value worker. Contains no expected answers or pass/fail logic."""

import contextlib
import json
import math
import resource
import runpy
import sys

TAGS = ("$tuple", "$set", "$frozenset", "$dict", "$complex", "$float")
MAX_DEPTH = 64


def encode_value(value, depth=0):
    if depth > MAX_DEPTH:
        raise ValueError("value nests too deeply")
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return {"$float": "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")}
    if isinstance(value, complex):
        return {"$complex": [encode_value(value.real, depth + 1), encode_value(value.imag, depth + 1)]}
    if isinstance(value, list):
        return [encode_value(v, depth + 1) for v in value]
    if isinstance(value, tuple):
        return {"$tuple": [encode_value(v, depth + 1) for v in value]}
    if isinstance(value, (set, frozenset)):
        tag = "$set" if isinstance(value, set) else "$frozenset"
        items = [encode_value(v, depth + 1) for v in value]
        return {tag: sorted(items, key=lambda v: json.dumps(v, sort_keys=True))}
    if isinstance(value, dict):
        if all(isinstance(k, str) for k in value) and not (len(value) == 1 and next(iter(value)) in TAGS):
            return {k: encode_value(v, depth + 1) for k, v in value.items()}
        return {"$dict": [[encode_value(k, depth + 1), encode_value(v, depth + 1)] for k, v in value.items()]}
    raise TypeError(f"unsupported value type {type(value).__name__}")


def decode_value(value, depth=0):
    if depth > MAX_DEPTH:
        raise ValueError("value nests too deeply")
    if isinstance(value, list):
        return [decode_value(v, depth + 1) for v in value]
    if not isinstance(value, dict):
        return value
    if len(value) == 1:
        tag, body = next(iter(value.items()))
        if tag == "$float":
            if body not in ("nan", "inf", "-inf"):
                raise ValueError("invalid $float")
            return float(body)
        if tag in TAGS and not isinstance(body, list):
            raise ValueError(f"invalid {tag}")
        if tag == "$tuple":
            return tuple(decode_value(v, depth + 1) for v in body)
        if tag == "$set":
            return {decode_value(v, depth + 1) for v in body}
        if tag == "$frozenset":
            return frozenset(decode_value(v, depth + 1) for v in body)
        if tag == "$complex":
            if len(body) != 2:
                raise ValueError("invalid $complex")
            return complex(decode_value(body[0], depth + 1), decode_value(body[1], depth + 1))
        if tag == "$dict":
            out = {}
            for pair in body:
                if not isinstance(pair, list) or len(pair) != 2:
                    raise ValueError("invalid $dict")
                out[decode_value(pair[0], depth + 1)] = decode_value(pair[1], depth + 1)
            return out
    return {k: decode_value(v, depth + 1) for k, v in value.items()}


def _limits(cpu, output):
    for kind, limit in [(resource.RLIMIT_CPU, cpu),
                        (resource.RLIMIT_AS, 512 * 1024 * 1024 if output > 64 * 1024 else 256 * 1024 * 1024),
                        (resource.RLIMIT_FSIZE, output),
                        (resource.RLIMIT_NOFILE, 32),
                        (resource.RLIMIT_NPROC, 1),
                        (resource.RLIMIT_CORE, 0)]:
        resource.setrlimit(kind, (limit, limit))


def _batch(request):
    results = []
    with contextlib.redirect_stdout(sys.stderr):
        namespace = runpy.run_path("/job/solution.py")
        function = namespace[request["function"]]
        for call in request["calls"]:
            try:
                value = function(*decode_value(call["args"]), **decode_value(call["kwargs"]))
                if call.get("truth"):
                    value = bool(value)
                results.append({"ok": encode_value(value)})
            except Exception as exc:
                results.append({"error": type(exc).__name__})
    sys.stdout.write(json.dumps({"values": results}, allow_nan=False))


def main():
    with open("/job/request.json", encoding="utf-8") as stream:
        request = json.load(stream)
    if "calls" in request:
        _limits(10, 8 * 1024 * 1024)
        _batch(request)
        return
    # Hard limits survive exec; dropping namespace capabilities prevents raising
    # them later. No writable filesystem or inherited parent result pipe exists.
    _limits(3, 64 * 1024)
    # Ordinary debug prints do not become protocol data. Direct writes to stdout
    # remain untrusted and must pass controller schema/value validation.
    with contextlib.redirect_stdout(sys.stderr):
        namespace = runpy.run_path("/job/solution.py")
        value = namespace[request["function"]](*request["args"], **request["kwargs"])
    sys.stdout.write(json.dumps({"value": value}, allow_nan=False))


if __name__ == "__main__":
    main()
