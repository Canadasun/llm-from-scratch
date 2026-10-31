"""GROUND-TRUTH FIXTURE — the SAFE counterparts. The scanner must report ZERO
findings here (false-positive check). Parsed by tests, never run.
"""
import ast
import hashlib
import json
import os
import subprocess

import requests
import yaml
from yaml import SafeLoader


def shell_list(host):
    subprocess.run(["ping", "-c", "1", host])          # list form, no shell


def safe_eval(expr):
    return ast.literal_eval(expr)                      # not eval/exec


def safe_json(data):
    return json.loads(data)                            # json, not pickle


def safe_yaml(data):
    return yaml.safe_load(data)                        # safe_load


def safe_yaml_loader(data):
    return yaml.load(data, Loader=SafeLoader)          # explicit SafeLoader


def strong_hash(pw):
    return hashlib.sha256(pw).hexdigest()              # SHA-256


def tls_on(url):
    return requests.get(url)                           # verification left on


API_KEY = os.environ["API_KEY"]                        # from env, not a literal
password = "changeme"                                  # placeholder, not a real secret


def run_app(app):
    app.run()                                          # no debug=True


def sql_param(cursor, uid):
    cursor.execute("SELECT * FROM users WHERE id = ?", (uid,))  # parameterised
