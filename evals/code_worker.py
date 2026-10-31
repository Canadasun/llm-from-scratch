"""Sandbox-only value worker. Contains no expected answers or pass/fail logic."""

import contextlib
import json
import resource
import runpy
import sys


def main():
    # Hard limits survive exec; dropping namespace capabilities prevents raising
    # them later. No writable filesystem or inherited parent result pipe exists.
    for kind, limit in [(resource.RLIMIT_CPU, 3),
                        (resource.RLIMIT_AS, 256 * 1024 * 1024),
                        (resource.RLIMIT_FSIZE, 64 * 1024),
                        (resource.RLIMIT_NOFILE, 32),
                        (resource.RLIMIT_NPROC, 1),
                        (resource.RLIMIT_CORE, 0)]:
        resource.setrlimit(kind, (limit, limit))
    with open("/job/request.json", encoding="utf-8") as stream:
        request = json.load(stream)
    # Ordinary debug prints do not become protocol data. Direct writes to stdout
    # remain untrusted and must pass controller schema/value validation.
    with contextlib.redirect_stdout(sys.stderr):
        namespace = runpy.run_path("/job/solution.py")
        value = namespace[request["function"]](*request["args"], **request["kwargs"])
    sys.stdout.write(json.dumps({"value": value}, allow_nan=False))


if __name__ == "__main__":
    main()
