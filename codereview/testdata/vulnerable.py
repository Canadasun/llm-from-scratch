"""GROUND-TRUTH FIXTURE — intentionally vulnerable. Parsed by tests, never run.

Each function contains exactly one issue the scanner must detect (FN check).
"""
import hashlib
import os
import pickle
import random
import subprocess
import tempfile

import requests
import yaml


def shell_from_input(host):
    subprocess.run("ping " + host, shell=True)        # PY-SHELL-INJECTION CWE-78


def shell_os_system(host):
    os.system("ping " + host)                          # PY-SHELL-INJECTION CWE-78


def dangerous_eval(user_input):
    return eval(user_input)                            # PY-DANGEROUS-EVAL CWE-95


def unpickle(data):
    return pickle.loads(data)                          # PY-INSECURE-DESERIALIZATION CWE-502


def unsafe_yaml(data):
    return yaml.load(data)                             # PY-INSECURE-DESERIALIZATION CWE-502


def weak_password_hash(pw):
    return hashlib.md5(pw).hexdigest()                 # PY-WEAK-HASH CWE-327


def no_tls_check(url):
    return requests.get(url, verify=False)             # PY-TLS-VERIFY-DISABLED CWE-295


API_KEY = "AKIA1234567890REALKEY"                      # PY-HARDCODED-SECRET CWE-798


def run_app(app):
    app.run(debug=True)                                # PY-FLASK-DEBUG CWE-489


def sql_concat(cursor, uid):
    cursor.execute("SELECT * FROM users WHERE id = " + uid)   # PY-SQL-INJECTION CWE-89


def insecure_tmp():
    return tempfile.mktemp()                           # PY-INSECURE-TEMPFILE CWE-377


def weak_random_token():
    session_token = random.getrandbits(256)            # PY-WEAK-RANDOM CWE-330
    return session_token


DEBUG = True                                           # PY-DJANGO-DEBUG CWE-489


def extract(tf):
    tf.extractall("/tmp/out")                          # PY-TARFILE-TRAVERSAL CWE-22


def make_env():
    return Environment(autoescape=False)               # PY-JINJA-AUTOESCAPE-OFF CWE-79


def serve_public(app):
    app.run(host="0.0.0.0")                            # PY-BIND-ALL-INTERFACES CWE-668
