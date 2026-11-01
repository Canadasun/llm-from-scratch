"""Trusted sandbox bootstrap: apply limits AFTER namespace setup, before exec.

Run by absolute path with isolated system Python. This file is mounted read-only
outside the writable repository. No repository imports or startup hooks run here.
"""
import math
import os
import resource
import sys


def main():
    seconds = math.ceil(float(sys.argv[1]))
    for kind, value in ((resource.RLIMIT_CPU, seconds),
                        (resource.RLIMIT_AS, 1024**3),
                        (resource.RLIMIT_NPROC, 64),
                        (resource.RLIMIT_NOFILE, 128),
                        (resource.RLIMIT_FSIZE, 8 * 1024**2),
                        (resource.RLIMIT_CORE, 0)):
        resource.setrlimit(kind, (value, value))
    command = sys.argv[2:]
    os.execvpe(command[0], command, os.environ)


if __name__ == "__main__":
    main()
