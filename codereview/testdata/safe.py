"""GROUND-TRUTH FIXTURE — the SAFE counterparts. The scanner must report ZERO
findings here (false-positive check). Parsed by tests, never run.
"""
import ast
import hashlib
import json
import os
import random
import secrets
import subprocess
import tempfile

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


def safe_tmp():
    return tempfile.mkstemp()                          # atomic, not mktemp


def secret_token():
    session_token = secrets.token_hex(32)              # secrets, not random
    return session_token


def non_secret_random():
    shuffle_index = random.randint(0, 9)               # not a secret value
    return shuffle_index


DEBUG = False                                          # not True
debug_mode = True                                      # name is not DEBUG


def extract_safe(tf, path):
    tf.extractall(path, filter="data")                 # filtered extract


def make_env_safe():
    return Environment(autoescape=True)                # autoescape on


def serve_local(app):
    app.run(host="127.0.0.1")                          # loopback, not 0.0.0.0
