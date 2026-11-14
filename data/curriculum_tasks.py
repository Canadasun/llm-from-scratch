from data.verified_tasks import Task, case


def task(family, split, spec, source, cases, mutation, domain="python"):
    return Task(family, split, spec, source, cases, mutation, domain, "data/curriculum_tasks.py")


TASKS = [
    task("tenant-scope-intersection", "train",
         "Implement can_read_record(record, principal). Both are dictionaries. Allow only when both have a "
         "nonempty string tenant_id, the tenant IDs are identical, principal.active is exactly True, and "
         "principal.scopes contains the exact string 'record:read'. Missing fields deny access. No wildcard bypass.",
         'def can_read_record(record, principal):\n    tenant = record.get("tenant_id")\n    scopes = principal.get("scopes")\n'
         '    return (isinstance(tenant, str) and bool(tenant) and tenant == principal.get("tenant_id")\n'
         '            and principal.get("active") is True and isinstance(scopes, list) and "record:read" in scopes)\n',
         [case([{"tenant_id": "a"}, {"tenant_id": "a", "active": True, "scopes": ["record:read"]}], True),
          case([{"tenant_id": "a"}, {"tenant_id": "b", "active": True, "scopes": ["record:read"]}], False),
          case([{}, {"active": True, "scopes": ["record:read"]}], False),
          case([{"tenant_id": "a"}, {"tenant_id": "a", "active": 1, "scopes": ["record:read"]}], False),
          case([{"tenant_id": "a"}, {"tenant_id": "a", "active": True, "scopes": ["*"]}], False),
          case([{"tenant_id": "a"}, {"tenant_id": "a", "active": True, "scopes": "record:read"}], False),
          case([{"tenant_id": ""}, {"tenant_id": "", "active": True, "scopes": ["record:read"]}], False)],
         ('tenant == principal.get("tenant_id")', 'True'), "security"),
    task("exact-grant-narrowing", "train",
         "Implement narrow_scopes(granted, requested). Both are lists of strings. Return sorted unique requested "
         "scopes if every requested string occurs exactly in granted. Otherwise raise ValueError. Treat '*' as "
         "an ordinary string, not a wildcard. Never add unrequested scopes or mutate either list.",
         'def narrow_scopes(granted, requested):\n    if not set(requested).issubset(set(granted)):\n'
         '        raise ValueError("scope escalation")\n    return sorted(set(requested))\n',
         [case([["a", "b"], ["b", "b"]], ["b"]), case([["a"], []], []),
          case([["a"], ["b"]], raises="ValueError"), case([["*"], ["read"]], raises="ValueError"),
          case([["b", "a"], ["b", "a"]], ["a", "b"])],
         ('set(requested).issubset(set(granted))', 'True'), "security"),
    task("tenant-owner-or-admin", "train",
         "Implement can_edit_document(document, actor). Require a nonempty string document.tenant equal to "
         "actor.tenant and actor.enabled exactly True. Then allow an exact 'admin' role or a nonempty string "
         "actor.id equal to document.owner. Missing values deny access; admins cannot cross tenants.",
         'def can_edit_document(document, actor):\n    tenant = document.get("tenant")\n'
         '    if not isinstance(tenant, str) or not tenant or tenant != actor.get("tenant") or actor.get("enabled") is not True:\n'
         '        return False\n    identifier = actor.get("id")\n'
         '    return actor.get("role") == "admin" or (isinstance(identifier, str) and bool(identifier) and identifier == document.get("owner"))\n',
         [case([{"tenant": "a", "owner": "u"}, {"tenant": "a", "id": "u", "enabled": True}], True),
          case([{"tenant": "a"}, {"tenant": "a", "role": "admin", "enabled": True}], True),
          case([{"tenant": "a"}, {"tenant": "b", "role": "admin", "enabled": True}], False),
          case([{"tenant": "a"}, {"tenant": "a", "enabled": True}], False),
          case([{"tenant": "a", "owner": "u"}, {"tenant": "a", "id": "u", "enabled": False}], False)],
         ('tenant != actor.get("tenant")', 'False'), "security"),
    task("token-validity-interval", "train",
         "Implement token_is_current(issued, expires, now, maximum_age). All inputs are integers. Return True "
         "only when maximum_age >= 0, issued <= now < expires, and now - issued <= maximum_age. "
         "A token is expired exactly at expires; future-issued tokens are invalid.",
         'def token_is_current(issued, expires, now, maximum_age):\n'
         '    return maximum_age >= 0 and issued <= now < expires and now - issued <= maximum_age\n',
         [case([10, 20, 15, 5], True), case([10, 20, 20, 20], False), case([16, 20, 15, 5], False),
          case([10, 30, 20, 9], False), case([10, 20, 10, 0], True), case([10, 20, 10, -1], False)],
         ('now < expires', 'now <= expires'), "security"),
    task("origin-tuple-validation", "train",
         "Implement origin_key(value). Parse a literal HTTP(S) origin with urllib.parse.urlsplit. Require a "
         "hostname, no username/password, no query or fragment, and a path that is empty or '/'. Accept only "
         "http and https. Reject ASCII controls and whitespace anywhere. Return [lowercase_scheme, lowercase_hostname, effective_port] using 80/443 if absent. "
         "Return None for invalid ports, non-origins or parse failures. Do not resolve hosts or contact a network.",
         'def origin_key(value):\n    from urllib.parse import urlsplit\n'
         '    if any(ord(c) <= 32 or ord(c) == 127 for c in value):\n        return None\n    try:\n        parsed = urlsplit(value)\n'
         '        if (parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username is not None\n'
         '                or parsed.password is not None or parsed.query or parsed.fragment or parsed.path not in {"", "/"}):\n'
         '            return None\n        port = parsed.port\n'
         '        return [parsed.scheme, parsed.hostname.lower(), port if port is not None else (443 if parsed.scheme == "https" else 80)]\n'
         '    except ValueError:\n        return None\n',
         [case(["https://Example.test"], ["https", "example.test", 443]),
          case(["http://example.test:8080/"], ["http", "example.test", 8080]),
          case(["https://u@example.test"], None), case(["https://example.test/path"], None),
          case(["https://example.test:99999"], None), case(["//example.test"], None), case(["https://example.test?q=1"], None),
          case(["https://exam\nple.test"], None)],
         ('parsed.username is not None', 'False'), "security"),
    task("password-input-byte-budget", "train",
         "Implement password_size_ok(password, minimum, maximum). password is a string and bounds are integers. "
         "Reject invalid bounds (minimum < 1 or maximum < minimum) with ValueError. Return whether the UTF-8 "
         "byte length, not character length, is within the inclusive bounds. This checks size only, not strength.",
         'def password_size_ok(password, minimum, maximum):\n    if minimum < 1 or maximum < minimum:\n'
         '        raise ValueError("bounds")\n    return minimum <= len(password.encode("utf-8")) <= maximum\n',
         [case(["abc", 3, 5], True), case(["éé", 1, 3], False), case(["", 1, 5], False),
          case(["abc", 0, 3], raises="ValueError"), case(["abc", 4, 3], raises="ValueError")],
         ('len(password.encode("utf-8"))', 'len(password)'), "security"),
    task("required-response-header-overlay", "train",
         "Implement enforce_headers(pairs, required). pairs is a list of string [key, value] headers; required "
         "is a dictionary with lowercase header names. Return a list retaining original headers except any "
         "whose lowercase key is in required, then append required's entries in sorted key order. Preserve "
         "non-required duplicates and do not mutate inputs.",
         'def enforce_headers(pairs, required):\n    kept = [[key, value] for key, value in pairs if key.lower() not in required]\n'
         '    return kept + [[key, required[key]] for key in sorted(required)]\n',
         [case([[], {}], []), case([[["X-Test", "old"], ["x-test", "other"]], {"x-test": "new"}], [["x-test", "new"]]),
          case([[["x", "1"], ["x", "2"]], {"z": "3", "a": "4"}], [["x", "1"], ["x", "2"], ["a", "4"], ["z", "3"]]),
          case([[["x", "1"]], {}], [["x", "1"]])], ('key.lower() not in required', 'key not in required'), "security"),
    task("literal-method-policy", "train",
         "Implement method_allowed(method, allowed). Inputs are a string and a list of uppercase method names. "
         "Accept only a nonempty ASCII uppercase A-Z method and exact membership in allowed. Reject whitespace, "
         "lowercase and nonletters; do not normalize the input or interpret wildcards.",
         'def method_allowed(method, allowed):\n'
         '    return bool(method) and all("A" <= c <= "Z" for c in method) and method in allowed\n',
         [case(["GET", ["GET"]], True), case(["get", ["GET"]], False), case([" GET", ["GET"]], False),
          case(["", [""]], False), case(["POST", ["GET", "*"]], False), case(["GET1", ["GET1"]], False)],
         ('all("A" <= c <= "Z" for c in method)', 'True'), "security"),
    task("single-positional-argv", "train",
         "Implement positional_argv(executable, value). The caller's trusted executable honors '--' as an option "
         "terminator. value is untrusted text. Reject an empty value or any ASCII control character (0-31 or 127) "
         "with ValueError. Return [executable, '--', value] without a shell or splitting value. Do not execute anything.",
         'def positional_argv(executable, value):\n'
         '    if not value or any(ord(c) < 32 or ord(c) == 127 for c in value):\n        raise ValueError("argument")\n'
         '    return [executable, "--", value]\n',
         [case(["tool", "-n"], ["tool", "--", "-n"]), case(["tool", "a; echo demo"], ["tool", "--", "a; echo demo"]),
          case(["tool", "two words"], ["tool", "--", "two words"]), case(["tool", ""], raises="ValueError"),
          case(["tool", "a\nb"], raises="ValueError")], ('[executable, "--", value]', '[executable, value]'), "security"),
    task("sqlite-column-whitelist", "train",
         "Implement user_lookup_query(column, value). Allow exactly id, name or email as column names, otherwise "
         "raise ValueError. Return [SQL, parameters], where SQL is 'SELECT id FROM users WHERE COLUMN = ?' with "
         "the approved column inserted, and parameters is a one-item list holding value unchanged. Never put "
         "value into SQL. This constructs a SQLite parameterized query; do not execute it.",
         'def user_lookup_query(column, value):\n    if column not in {"id", "name", "email"}:\n'
         '        raise ValueError("column")\n    return ["SELECT id FROM users WHERE " + column + " = ?", [value]]\n',
         [case(["email", "x@example.test"], ["SELECT id FROM users WHERE email = ?", ["x@example.test"]]),
          case(["id", 0], ["SELECT id FROM users WHERE id = ?", [0]]),
          case(["name", "x' OR '1'='1"], ["SELECT id FROM users WHERE name = ?", ["x' OR '1'='1"]]),
          case(["email OR 1=1", "x"], raises="ValueError"), case(["ID", 1], raises="ValueError")],
         ('column not in {"id", "name", "email"}', 'False'), "security"),
    task("literal-relative-segments", "train",
         "Implement relative_segments(path). This is only a lexical POSIX-path check, not filesystem confinement. "
         "Reject empty paths, leading '/', backslashes, NUL, and '.' or '..' components with ValueError. "
         "Discard empty components from repeated/trailing slashes and return the rest as a list. Do not URL-decode.",
         'def relative_segments(path):\n    if not path or path.startswith("/") or "\\\\" in path or "\\x00" in path:\n'
         '        raise ValueError("path")\n    parts = [part for part in path.split("/") if part]\n'
         '    if any(part in {".", ".."} for part in parts):\n        raise ValueError("path")\n    return parts\n',
         [case(["a//b/"], ["a", "b"]), case(["/a"], raises="ValueError"), case(["a/../b"], raises="ValueError"),
          case(["a/./b"], raises="ValueError"), case(["a\\b"], raises="ValueError"), case([""], raises="ValueError")],
         ('{".", ".."}', '{"."}'), "security"),
    task("ascii-attachment-name", "train",
         "Implement attachment_name_ok(name). Accept only 1..64 ASCII characters from letters, digits, '_', '-' "
         "and '.', and require the first character not to be '.'. Return a boolean. This is a restricted name "
         "policy, not a filesystem authorization or a check of platform-reserved names.",
         'def attachment_name_ok(name):\n    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-."\n'
         '    return 1 <= len(name) <= 64 and not name.startswith(".") and all(c in allowed for c in name)\n',
         [case(["report-1.txt"], True), case([".env"], False), case(["a/b"], False),
          case(["é.txt"], False), case(["a" * 65], False), case([""], False)],
         ('not name.startswith(".")', 'True'), "security"),
    task("restricted-local-redirect", "train",
         "Implement local_redirect_path(value). Return value unchanged only if it starts with exactly one '/', "
         "contains only ASCII letters, digits, '/', '_', '-' and '.', and no component equals '.' or '..'. "
         "Return None otherwise. Empty components beyond the first are permitted. No query strings, percent "
         "escapes, backslashes, credentials, schemes or fragments are supported.",
         'def local_redirect_path(value):\n    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/_-."\n'
         '    if not value.startswith("/") or value.startswith("//") or any(c not in alphabet for c in value):\n'
         '        return None\n    if any(part in {".", ".."} for part in value.split("/")):\n        return None\n'
         '    return value\n',
         [case(["/account/home"], "/account/home"), case(["/"], "/"), case(["//example.test"], None),
          case(["/%2fexample.test"], None), case(["/a/../b"], None), case(["/x?q=1"], None)],
         ('value.startswith("//")', 'False'), "security"),
    task("strict-positive-budget-integer", "train",
         "Implement require_budget(value, maximum). maximum is a positive integer. Accept value only if its "
         "type is exactly int, it is at least 1 and it does not exceed maximum; return it. Reject booleans, "
         "floats, strings and out-of-range integers with ValueError.",
         'def require_budget(value, maximum):\n    if type(value) is not int or not 1 <= value <= maximum:\n'
         '        raise ValueError("budget")\n    return value\n',
         [case([1, 10], 1), case([10, 10], 10), case([True, 10], raises="ValueError"),
          case([1.0, 10], raises="ValueError"), case([0, 10], raises="ValueError"), case([11, 10], raises="ValueError")],
         ('type(value) is not int', 'not isinstance(value, int)'), "security"),
    task("upload-budget-accounting", "train",
         "Implement can_accept_upload(used, incoming, limit). All inputs may be arbitrary JSON values. Return "
         "False unless all three have type exactly int and are nonnegative. Otherwise allow exactly when "
         "used + incoming <= limit. Return only a boolean; do not mutate or reserve a quota.",
         'def can_accept_upload(used, incoming, limit):\n'
         '    if any(type(value) is not int or value < 0 for value in (used, incoming, limit)):\n        return False\n'
         '    return used + incoming <= limit\n',
         [case([3, 2, 5], True), case([3, 3, 5], False), case([-1, 1, 5], False),
          case([False, 1, 5], False), case(["0", 1, 5], False), case([0, 0, 0], True)],
         ('used + incoming <= limit', 'incoming <= limit'), "security"),
    task("opaque-token-alphabet", "train",
         "Implement token_shape_ok(text). Return True only for a string of 16..128 characters from the ASCII "
         "base64url alphabet (A-Z, a-z, 0-9, '_' and '-'), without padding. Accept arbitrary JSON values as input; "
         "return False for nonstrings. This is a format check only and does not authenticate a token.",
         'def token_shape_ok(text):\n    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-"\n'
         '    return isinstance(text, str) and 16 <= len(text) <= 128 and all(c in alphabet for c in text)\n',
         [case(["a" * 16], True), case(["a" * 15], False), case(["a" * 129], False),
          case(["a" * 15 + "="], False), case([None], False), case(["é" * 16], False)],
         ('16 <= len(text) <= 128', '1 <= len(text) <= 128'), "security"),
    task("query-secret-preservation", "train",
         "Implement redact_query_pairs(pairs, names). pairs is a list of string [key, value] pairs and names is "
         "a list of sensitive names. Return new pairs, preserving keys, order and duplicates, replacing values "
         "whose key matches a name ignoring case with '[redacted]'. Match whole keys, not substrings.",
         'def redact_query_pairs(pairs, names):\n    hidden = {name.lower() for name in names}\n'
         '    return [[key, "[redacted]" if key.lower() in hidden else value] for key, value in pairs]\n',
         [case([[], ["secret"]], []), case([[["Token", "a"], ["token", "b"]], ["TOKEN"]], [["Token", "[redacted]"], ["token", "[redacted]"]]),
          case([[["token_hint", "ok"]], ["token"]], [["token_hint", "ok"]]),
          case([[["q", "x"]], []], [["q", "x"]])], ('key.lower() in hidden', 'key in hidden'), "security"),
    task("session-cookie-policy", "train",
         "Implement session_cookie_options(value, ttl). ttl must have type exactly int in 1..86400 inclusive, "
         "else raise ValueError. Return a new dictionary with value unchanged, secure=True, httponly=True, "
         "samesite='Lax', path='/' and max_age=ttl. This returns cookie options only, not a complete session system.",
         'def session_cookie_options(value, ttl):\n    if type(ttl) is not int or not 1 <= ttl <= 86400:\n'
         '        raise ValueError("ttl")\n    return {"value": value, "secure": True, "httponly": True, "samesite": "Lax", "path": "/", "max_age": ttl}\n',
         [case(["opaque", 60], {"value": "opaque", "secure": True, "httponly": True, "samesite": "Lax", "path": "/", "max_age": 60}),
          case(["opaque", 0], raises="ValueError"), case(["opaque", 86401], raises="ValueError"),
          case(["opaque", True], raises="ValueError"), case(["opaque", "60"], raises="ValueError")],
         ('"httponly": True', '"httponly": False'), "security"),
    task("dependency-exact-pin-parser", "train",
         "Implement parse_exact_pin(line). Strip surrounding whitespace. Accept only a package name of ASCII "
         "letters, digits, '-' or '_' followed by '==' and exactly three nonempty dot-separated ASCII integer "
         "components. Return [lowercase_name, version] preserving version digits, or None. Reject extras, "
         "URLs, comments, ranges and spaces inside the pin. This is a syntax parser, not a package trust decision.",
         'def parse_exact_pin(line):\n    import re\n    match = re.fullmatch(r"([A-Za-z0-9_-]+)==([0-9]+\\.[0-9]+\\.[0-9]+)", line.strip())\n'
         '    return [match.group(1).lower(), match.group(2)] if match else None\n',
         [case([" Tool-X==1.2.3 "], ["tool-x", "1.2.3"]), case(["x>=1.2.3"], None),
          case(["x==1.2"], None), case(["x==1.2.3 # note"], None), case(["x[y]==1.2.3"], None),
          case(["x==01.2.3"], ["x", "01.2.3"])], ('re.fullmatch', 're.match')),
    task("required-string-setting", "train",
         "Implement required_setting(settings, key). Return the stripped value at key only if it is a string "
         "and remains nonempty after stripping. Otherwise raise ValueError. Do not read environment variables, "
         "coerce other types or mutate settings.",
         'def required_setting(settings, key):\n    value = settings.get(key)\n'
         '    if not isinstance(value, str) or not value.strip():\n        raise ValueError("required setting")\n'
         '    return value.strip()\n',
         [case([{"name": " value "}, "name"], "value"), case([{}, "name"], raises="ValueError"),
          case([{"name": " "}, "name"], raises="ValueError"), case([{"name": 0}, "name"], raises="ValueError"),
          case([{"name": "0"}, "name"], "0")], ('return value.strip()', 'return value')),
    task("nested-dictionary-path", "train",
         "Implement config_at(root, keys, default). Follow each string in keys through dictionaries only, "
         "returning default when a key is missing or an intermediate value is not a dictionary. Preserve "
         "existing None values. An empty keys list returns root. Do not mutate inputs.",
         'def config_at(root, keys, default):\n    value = root\n    for key in keys:\n'
         '        if not isinstance(value, dict) or key not in value:\n            return default\n'
         '        value = value[key]\n    return value\n',
         [case([{"a": {"b": 2}}, ["a", "b"], 0], 2), case([{"a": None}, ["a"], 0], None),
          case([{"a": None}, ["a", "b"], 0], 0), case([{"a": 1}, [], 0], {"a": 1}),
          case([{}, ["a"], "missing"], "missing")], ('return value\n', 'return value or default\n')),
    task("append-only-config-lists", "train",
         "Implement merge_list_settings(parts). parts is a list of dictionaries mapping strings to lists. "
         "Return a dictionary concatenating lists per key in part order, retaining duplicates. Missing keys "
         "start empty. Do not mutate any input list or dictionary.",
         'def merge_list_settings(parts):\n    result = {}\n    for part in parts:\n        for key, values in part.items():\n'
         '            result[key] = result.get(key, []) + list(values)\n    return result\n',
         [case([[]], {}), case([[{"x": [1]}, {"x": [2, 1]}]], {"x": [1, 2, 1]}),
          case([[{"x": []}, {"y": [3]}]], {"x": [], "y": [3]}), case([[{"x": [0]}]], {"x": [0]})],
         ('result.get(key, []) + list(values)', 'list(values)')),
    task("strict-hex-byte-decoder", "train",
         "Implement decode_hex_bytes(text). Accept only an even number of ASCII hexadecimal digits, including "
         "an empty string. Return decoded byte values as a list of integers. Reject whitespace, '0x', odd "
         "length or other characters with ValueError.",
         'def decode_hex_bytes(text):\n    if len(text) % 2 or any(c not in "0123456789abcdefABCDEF" for c in text):\n'
         '        raise ValueError("hex")\n    return list(bytes.fromhex(text))\n',
         [case(["00Ffa1"], [0, 255, 161]), case([""], []), case(["0"], raises="ValueError"),
          case(["aa bb "], raises="ValueError"), case(["0xff"], raises="ValueError")],
         ('any(c not in "0123456789abcdefABCDEF" for c in text)', 'False')),
    task("utc-log-timestamp", "train",
         "Implement log_epoch_seconds(text). Parse exactly the shape YYYY-MM-DDTHH:MM:SSZ using ASCII digits "
         "and zero-padded components. Return integer seconds since the Unix epoch in UTC. Raise ValueError "
         "for malformed or impossible dates. Never use the host's local timezone.",
         'def log_epoch_seconds(text):\n    import re\n    from datetime import datetime, timezone\n'
         '    if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", text):\n'
         '        raise ValueError("timestamp")\n'
         '    value = datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)\n'
         '    return int(value.timestamp())\n',
         [case(["1970-01-01T00:00:00Z"], 0), case(["1970-01-02T00:00:00Z"], 86400),
          case(["2000-02-29T00:00:00Z"], 951782400), case(["2001-02-29T00:00:00Z"], raises="ValueError"),
          case(["1970-1-01T00:00:00Z"], raises="ValueError")],
         ('if not re.fullmatch', 'if False and not re.fullmatch')),
    task("dependency-ready-frontier", "train",
         "Implement ready_steps(dependencies, completed). dependencies maps step names to lists of prerequisite "
         "names. completed is a list of already completed names. Return sorted step names that are not completed "
         "and whose prerequisites are all completed. Unlisted prerequisites are not implicitly complete. "
         "Cycles remain blocked; do not recursively execute or schedule anything.",
         'def ready_steps(dependencies, completed):\n    done = set(completed)\n'
         '    return sorted(name for name, needs in dependencies.items() if name not in done and all(need in done for need in needs))\n',
         [case([{"b": ["a"], "a": []}, []], ["a"]), case([{"a": [], "b": ["a"]}, ["a"]], ["b"]),
          case([{"a": ["b"], "b": ["a"]}, []], []), case([{"a": ["external"]}, []], []),
          case([{}, []], [])], ('all(need in done for need in needs)', 'any(need in done for need in needs)')),
    task("canonical-query-pair-order", "train",
         "Implement canonical_pairs(pairs). Return a new list of string [key, value] pairs sorted first by key "
         "then value, retaining duplicate pairs and case. Do not combine values or mutate the input.",
         'def canonical_pairs(pairs):\n    return [[key, value] for key, value in sorted(pairs, key=lambda pair: (pair[0], pair[1]))]\n',
         [case([[]], []), case([[["b", "0"], ["a", "2"], ["a", "1"]]], [["a", "1"], ["a", "2"], ["b", "0"]]),
          case([[["a", "1"], ["a", "1"]]], [["a", "1"], ["a", "1"]]),
          case([[["a", "0"], ["A", "1"]]], [["A", "1"], ["a", "0"]])],
         ('(pair[0], pair[1])', '(pair[0],)')),
    task("json-pointer-token-decoding", "train",
         "Implement pointer_token(token). Decode JSON Pointer token escapes: '~1' becomes '/' and '~0' becomes '~'. "
         "Reject every other '~' escape, including a terminal '~', with ValueError. Scan left to right so '~01' "
         "becomes '~1', not '/'. Do not split the token or modify ordinary characters.",
         'def pointer_token(token):\n    result = []\n    index = 0\n    while index < len(token):\n'
         '        if token[index] != "~":\n            result.append(token[index])\n            index += 1\n            continue\n'
         '        if index + 1 >= len(token) or token[index + 1] not in "01":\n            raise ValueError("escape")\n'
         '        result.append("~" if token[index + 1] == "0" else "/")\n        index += 2\n    return "".join(result)\n',
         [case(["a~1b"], "a/b"), case(["~01"], "~1"), case([""], ""),
          case(["~2"], raises="ValueError"), case(["end~"], raises="ValueError"), case(["~0~1"], "~/")],
         ('token[index + 1] == "0"', 'token[index + 1] == "1"')),
    task("bounded-list-batches", "train",
         "Implement make_batches(items, size). items is a list; size must have type exactly int and be positive, "
         "else raise ValueError. Return consecutive sublists of at most size items, including the final short "
         "batch. An empty input gives an empty list. Do not mutate items.",
         'def make_batches(items, size):\n    if type(size) is not int or size <= 0:\n        raise ValueError("size")\n'
         '    return [items[start:start + size] for start in range(0, len(items), size)]\n',
         [case([[1, 2, 3], 2], [[1, 2], [3]]), case([[], 2], []), case([[1], 3], [[1]]),
          case([[1], 0], raises="ValueError"), case([[1], True], raises="ValueError")],
         ('type(size) is not int', 'not isinstance(size, int)')),
    task("severity-prefix-parser", "train",
         "Implement severity_message(line). Partition at the first ':'. If the stripped prefix is exactly "
         "DEBUG, INFO, WARNING or ERROR, return [prefix, stripped_remainder]; otherwise return None. "
         "The colon is required; keep additional colons inside the message. Comparisons are case-sensitive.",
         'def severity_message(line):\n    prefix, separator, message = line.partition(":")\n    prefix = prefix.strip()\n'
         '    if separator and prefix in {"DEBUG", "INFO", "WARNING", "ERROR"}:\n        return [prefix, message.strip()]\n'
         '    return None\n',
         [case([" INFO: x: y "], ["INFO", "x: y"]), case(["ERROR:"], ["ERROR", ""]),
          case(["info: x"], None), case(["INFO"], None), case(["OTHER: x"], None)],
         ('if separator and prefix in', 'if prefix in')),
    task("utf8-prefix-byte-limit", "train",
         "Implement utf8_prefix(text, byte_limit). byte_limit is a nonnegative integer; raise ValueError if "
         "negative. Encode text as UTF-8, keep at most byte_limit bytes, and decode while discarding only an "
         "incomplete final character. Return a string without exceeding that byte budget.",
         'def utf8_prefix(text, byte_limit):\n    if byte_limit < 0:\n        raise ValueError("limit")\n'
         '    return text.encode("utf-8")[:byte_limit].decode("utf-8", errors="ignore")\n',
         [case(["abéz", 3], "ab"), case(["abéz", 4], "abé"), case(["x", 0], ""),
          case(["", 5], ""), case(["x", -1], raises="ValueError")],
         ('text.encode("utf-8")[:byte_limit].decode("utf-8", errors="ignore")', 'text[:byte_limit]')),
    task("cidr-literal-membership", "validation",
         "Implement address_in_networks(address, networks). Use ipaddress.ip_address and ip_network(strict=False). "
         "Return whether a literal IP address belongs to any same-version network. Invalid address or network "
         "strings raise ValueError, even if another network already matches. Never resolve DNS names.",
         'def address_in_networks(address, networks):\n    import ipaddress\n    ip = ipaddress.ip_address(address)\n'
         '    parsed = [ipaddress.ip_network(network, strict=False) for network in networks]\n'
         '    return any(ip.version == network.version and ip in network for network in parsed)\n',
         [case(["192.0.2.5", ["192.0.2.0/24"]], True), case(["198.51.100.1", ["192.0.2.0/24"]], False),
          case(["2001:db8::1", ["2001:db8::/32"]], True), case(["192.0.2.1", []], False),
          case(["example.test", ["192.0.2.0/24"]], raises="ValueError"),
          case(["192.0.2.5", ["192.0.2.0/24", "bad"]], raises="ValueError")],
         ('ip in network', 'ip not in network'), "security"),
    task("hmac-sha256-message-check", "validation",
         "Implement verify_message_hmac(body, key, signature). body and key are strings encoded as UTF-8. "
         "signature may be any JSON value. Return False unless signature is exactly 64 lowercase ASCII hex "
         "characters; otherwise compare it to HMAC-SHA256(key, body) with hmac.compare_digest. Return a boolean.",
         'def verify_message_hmac(body, key, signature):\n    import hashlib\n    import hmac\n    import re\n'
         '    if not isinstance(signature, str) or not re.fullmatch(r"[0-9a-f]{64}", signature):\n        return False\n'
         '    expected = hmac.new(key.encode("utf-8"), body.encode("utf-8"), hashlib.sha256).hexdigest()\n'
         '    return hmac.compare_digest(expected, signature)\n',
         [case(["message", "demo-key", "4ac90015300d3b4633ef030781f8582ae7ac818d373c5c11e0fcebcf26bcb4c5"], True),
          case(["changed", "demo-key", "4ac90015300d3b4633ef030781f8582ae7ac818d373c5c11e0fcebcf26bcb4c5"], False),
          case(["", "demo-key", "8a9016cda075fc942ab4ae3b189fcb2011cfdc22916a63c5d2684daca8f1f026"], True),
          case(["message", "demo-key", "0" * 64], False), case(["message", "demo-key", None], False)],
         ('hmac.compare_digest(expected, signature)', 'True'), "security"),
    task("permission-write-bit-removal", "validation",
         "Implement private_write_mode(mode). mode is an integer permission word. Keep only its low nine "
         "permission bits, remove group-write and other-write (0o022), and return the result as an integer. "
         "Do not access or change any files; this only calculates a mode.",
         'def private_write_mode(mode):\n    return mode & 0o777 & ~0o022\n',
         [case([0o777], 0o755), case([0o644], 0o644), case([0o666], 0o644),
          case([0o100600], 0o600), case([0], 0)], ('~0o022', '~0o002'), "security"),
    task("tab-stop-expansion", "validation",
         "Implement expand_log_tabs(text, width). width is a positive integer, else raise ValueError. Return "
         "text with tabs expanded to the next multiple-of-width column, resetting the column after '\\n' or "
         "'\\r', exactly as str.expandtabs(width). Do not count every tab as a fixed number of spaces.",
         'def expand_log_tabs(text, width):\n    if width <= 0:\n        raise ValueError("width")\n'
         '    return text.expandtabs(width)\n',
         [case(["a\tb", 4], "a   b"), case(["\t", 2], "  "), case(["abc\n\tx", 4], "abc\n    x"),
          case(["", 3], ""), case(["x", 0], raises="ValueError")],
         ('text.expandtabs(width)', 'text.replace("\\t", " " * width)')),
    task("csv-fixed-width-records", "validation",
         "Implement csv_records(text, width). Parse comma-separated records using csv.reader and io.StringIO. "
         "Skip records that csv.reader returns as an empty list. Every other record must have exactly width "
         "fields, otherwise raise ValueError. Preserve field whitespace and quoted commas. Return lists of strings.",
         'def csv_records(text, width):\n    import csv\n    import io\n    records = []\n'
         '    for row in csv.reader(io.StringIO(text)):\n        if not row:\n            continue\n'
         '        if len(row) != width:\n            raise ValueError("record width")\n        records.append(row)\n'
         '    return records\n',
         [case(['a,"b,c"\nx,y\n', 2], [["a", "b,c"], ["x", "y"]]), case(["\n", 2], []),
          case(["a, b\n", 2], [["a", " b"]]), case(["a,b,c\n", 2], raises="ValueError"), case(["", 1], [])],
         ('len(row) != width', 'len(row) < width')),
    task("explicit-import-partition", "validation",
         "Implement partition_imports(names, standard). names and standard are lists of module names. "
         "Return {'relative': [...], 'standard': [...], 'third_party': [...]} with each category sorted and "
         "deduplicated. Names beginning '.' are relative; otherwise the first dot-separated component decides "
         "standard membership. Do not import modules or inspect installed packages.",
         'def partition_imports(names, standard):\n    result = {"relative": set(), "standard": set(), "third_party": set()}\n'
         '    for name in names:\n        category = "relative" if name.startswith(".") else ("standard" if name.split(".", 1)[0] in standard else "third_party")\n'
         '        result[category].add(name)\n    return {key: sorted(values) for key, values in result.items()}\n',
         [case([[], []], {"relative": [], "standard": [], "third_party": []}),
          case([["os.path", ".local", "flask", "os.path"], ["os"]], {"relative": [".local"], "standard": ["os.path"], "third_party": ["flask"]}),
          case([["package.sub"], []], {"relative": [], "standard": [], "third_party": ["package.sub"]}),
          case([["..a", ".a"], ["a"]], {"relative": ["..a", ".a"], "standard": [], "third_party": []})],
         ('name.split(".", 1)[0] in standard', 'name in standard')),
    task("exact-decimal-string-sum", "validation",
         "Implement add_decimal_strings(left, right). Inputs are optional '-' then 1..28 ASCII digits, optionally "
         "followed by '.' and 1..28 ASCII digits. Reject other forms with ValueError. Add exactly using Decimal "
         "in a local context of precision 80. Return fixed-point notation without unnecessary fractional zeros "
         "or a terminal dot; canonicalize negative zero as '0'. Do not use binary float arithmetic.",
         'def add_decimal_strings(left, right):\n    import re\n    from decimal import Decimal, localcontext\n'
         '    pattern = r"-?[0-9]{1,28}(?:\\.[0-9]{1,28})?"\n'
         '    if not re.fullmatch(pattern, left) or not re.fullmatch(pattern, right):\n        raise ValueError("decimal")\n'
         '    with localcontext() as context:\n        context.prec = 80\n        value = Decimal(left) + Decimal(right)\n'
         '    if not value:\n        return "0"\n    text = format(value, "f")\n'
         '    return text.rstrip("0").rstrip(".") if "." in text else text\n',
         [case(["0.1", "0.2"], "0.3"), case(["100", "0"], "100"), case(["-0.0", "0.00"], "0"),
          case(["-1.20", "0.2"], "-1"), case(["1e3", "0"], raises="ValueError"), case([".2", "0"], raises="ValueError")],
         ('Decimal(left) + Decimal(right)', 'Decimal(str(float(left) + float(right)))')),
    task("duplicate-json-key-rejection", "validation",
         "Implement load_unique_json(text). Use json.loads, but reject duplicate keys in any JSON object with "
         "ValueError, including nested objects. Repeated values in arrays are allowed. Return the parsed JSON "
         "value. Ordinary JSON parse errors may propagate as ValueError subclasses.",
         'def load_unique_json(text):\n    import json\n    def unique(pairs):\n        result = {}\n'
         '        for key, value in pairs:\n            if key in result:\n                raise ValueError("duplicate key")\n'
         '            result[key] = value\n        return result\n    return json.loads(text, object_pairs_hook=unique)\n',
         [case(['{"x":1}'], {"x": 1}), case(['{"x":1,"x":2}'], raises="ValueError"),
          case(['{"a":{"x":1,"x":2}}'], raises="ValueError"), case(['[1,1]'], [1, 1]), case(['{}'], {})],
         ('json.loads(text, object_pairs_hook=unique)', 'json.loads(text)')),
    task("strict-percent-byte-sequence", "validation",
         "Implement percent_bytes(text). text must be ASCII. Scan left to right: every '%' must be followed "
         "by two ASCII hexadecimal digits and represents one byte. Ordinary ASCII characters represent their "
         "byte value, including '+' unchanged. Return integer byte values as a list; malformed sequences or "
         "non-ASCII input raise ValueError. Do not decode the bytes as Unicode.",
         'def percent_bytes(text):\n    if not text.isascii():\n        raise ValueError("ASCII")\n'
         '    result = []\n    index = 0\n    while index < len(text):\n        if text[index] == "%":\n'
         '            part = text[index + 1:index + 3]\n            if len(part) != 2 or any(c not in "0123456789abcdefABCDEF" for c in part):\n'
         '                raise ValueError("escape")\n            result.append(int(part, 16))\n            index += 3\n'
         '        else:\n            result.append(ord(text[index]))\n            index += 1\n    return result\n',
         [case(["a%2Fb+"], [97, 47, 98, 43]), case(["%00"], [0]), case([""], []),
          case(["%0"], raises="ValueError"), case(["%GG"], raises="ValueError"), case(["é"], raises="ValueError")],
         ('ord(text[index])', '32 if text[index] == "+" else ord(text[index])')),
    task("html-text-and-attribute-escaping", "test",
         "Implement escape_html_value(text). Return html.escape(text, quote=True), escaping ampersands, '<', '>', "
         "double quotes and apostrophes. Do not trust existing entity-looking text or mark it safe. "
         "This is HTML text/quoted-attribute escaping, not JavaScript or URL-context escaping.",
         'def escape_html_value(text):\n    import html\n    return html.escape(text, quote=True)\n',
         [case(['<a x="y">'], '&lt;a x=&quot;y&quot;&gt;'), case(["it's & ok"], "it&#x27;s &amp; ok"),
          case(["&lt;"], "&amp;lt;"), case(["café"], "café"), case([""], "")],
         ('quote=True', 'quote=False'), "security"),
    task("json-container-depth-budget", "test",
         "Implement within_json_depth(value, maximum). value is an acyclic JSON value. A root dictionary or "
         "list has container depth 1; each directly nested dictionary/list increases depth by 1. Scalars do not "
         "increase depth. Return False for a negative maximum or any container depth greater than maximum; "
         "otherwise True. Traverse iteratively; do not stringify the value.",
         'def within_json_depth(value, maximum):\n    if maximum < 0:\n        return False\n'
         '    pending = [(value, 0)]\n    while pending:\n        item, parent_depth = pending.pop()\n'
         '        if isinstance(item, (dict, list)):\n            depth = parent_depth + 1\n'
         '            if depth > maximum:\n                return False\n'
         '            children = item.values() if isinstance(item, dict) else item\n'
         '            pending.extend((child, depth) for child in children)\n    return True\n',
         [case([1, 0], True), case([{}, 0], False), case([{"a": [1]}, 1], False),
          case([{"a": [1]}, 2], True), case([[], 1], True), case([None, -1], False)],
         ('depth > maximum', 'depth >= maximum'), "security"),
    task("archive-expansion-policy", "test",
         "Implement archive_entry_allowed(compressed, expanded, maximum, ratio_limit). All inputs must have "
         "type exactly int. compressed, expanded and maximum must be nonnegative; ratio_limit must be positive. "
         "Reject expanded > maximum. For compressed == 0, allow only expanded == 0; otherwise allow expanded "
         "<= compressed * ratio_limit. Return False for any invalid input. This is metadata policy only.",
         'def archive_entry_allowed(compressed, expanded, maximum, ratio_limit):\n'
         '    if any(type(v) is not int for v in (compressed, expanded, maximum, ratio_limit)):\n        return False\n'
         '    if min(compressed, expanded, maximum) < 0 or ratio_limit <= 0 or expanded > maximum:\n        return False\n'
         '    return expanded == 0 if compressed == 0 else expanded <= compressed * ratio_limit\n',
         [case([10, 100, 200, 10], True), case([10, 101, 200, 10], False), case([10, 100, 99, 10], False),
          case([0, 1, 100, 10], False), case([0, 0, 0, 10], True), case([True, 1, 10, 10], False)],
         ('expanded > maximum', 'False'), "security"),
    task("ascii-python-identifier", "test",
         "Implement ascii_identifier_ok(value). Accept arbitrary JSON values. Return True only if value is a "
         "nonempty ASCII string satisfying str.isidentifier() and is not a Python keyword. Return False otherwise. "
         "Do not normalize or transliterate Unicode identifiers.",
         'def ascii_identifier_ok(value):\n    import keyword\n'
         '    return isinstance(value, str) and value.isascii() and value.isidentifier() and not keyword.iskeyword(value)\n',
         [case(["name_1"], True), case(["class"], False), case(["é"], False), case(["1name"], False),
          case([None], False), case([""], False)], ('value.isascii()', 'True'), "security"),
    task("finite-json-numeric-value", "test",
         "Implement finite_number(value). Accept arbitrary JSON values. Return True only if its type is exactly "
         "int, or exactly float with math.isfinite(value). Booleans, None, strings and collections are False. "
         "Do not coerce values or convert arbitrarily large integers to float.",
         'def finite_number(value):\n    import math\n'
         '    return type(value) is int or (type(value) is float and math.isfinite(value))\n',
         [case([1], True), case([1.25], True), case([True], False), case([None], False),
          case(["1"], False), case([10 ** 100], True)],
         ('type(value) is int', 'isinstance(value, int)')),
    task("monotonic-event-index", "test",
         "Implement first_time_reversal(timestamps). timestamps is a list of integers in event order. "
         "Return the zero-based index of the first item strictly less than its immediate predecessor, or None "
         "if there is none. Equal adjacent times are allowed. Do not sort or mutate the input.",
         'def first_time_reversal(timestamps):\n    for index in range(1, len(timestamps)):\n'
         '        if timestamps[index] < timestamps[index - 1]:\n            return index\n    return None\n',
         [case([[1, 1, 2]], None), case([[1, 3, 2, 0]], 2), case([[]], None), case([[2]], None), case([[2, 1]], 1)],
         ('timestamps[index] < timestamps[index - 1]', 'timestamps[index] <= timestamps[index - 1]')),
    task("single-byte-range-parser", "test",
         "Implement single_byte_range(header, size). size is a nonnegative integer. Accept only ASCII syntax "
         "'bytes=START-END' or 'bytes=START-' with nonempty decimal START, no spaces or signs. Return inclusive "
         "[start, min(end, size-1)], using size-1 for omitted end. Return None for invalid syntax, size zero, "
         "start >= size, end < start, suffix or multiple ranges. Do not support other range units.",
         'def single_byte_range(header, size):\n    import re\n    match = re.fullmatch(r"bytes=([0-9]+)-([0-9]*)", header)\n'
         '    if not match or size <= 0:\n        return None\n    start = int(match.group(1))\n'
         '    end = int(match.group(2)) if match.group(2) else size - 1\n'
         '    if start >= size or end < start:\n        return None\n    return [start, min(end, size - 1)]\n',
         [case(["bytes=0-2", 10], [0, 2]), case(["bytes=2-", 5], [2, 4]), case(["bytes=0-99", 5], [0, 4]),
          case(["bytes=5-", 5], None), case(["bytes=-2", 5], None), case(["bytes=2-1", 5], None),
          case(["bytes=0-0", 0], None)], ('min(end, size - 1)', 'end')),
    task("normalized-key-collision", "test",
         "Implement normalized_keys(names). names is a list of strings. Normalize each with Unicode NFKC then "
         "casefold(). Return the normalized list in original order, but raise ValueError if any two results "
         "are equal. Do not remove whitespace, transliterate or silently merge duplicates.",
         'def normalized_keys(names):\n    import unicodedata\n    result = []\n    seen = set()\n'
         '    for name in names:\n        key = unicodedata.normalize("NFKC", name).casefold()\n'
         '        if key in seen:\n            raise ValueError("collision")\n        seen.add(key)\n        result.append(key)\n'
         '    return result\n',
         [case([["Ａ", "b"]], ["a", "b"]), case([["Ｋ", "k"]], raises="ValueError"),
          case([["Straße", "STRASSE"]], raises="ValueError"), case([[]], []), case([[" a", "a"]], [" a", "a"])],
         ('unicodedata.normalize("NFKC", name).casefold()', 'name.casefold()')),
    task("canonical-json-content-digest", "test",
         "Implement canonical_json_digest(value). Serialize the JSON value with json.dumps(sort_keys=True, "
         "separators=(',', ':'), ensure_ascii=True, allow_nan=False). UTF-8 encode the serialization and return "
         "its SHA-256 hexadecimal digest. Do not include a newline, use repr(), or rely on dictionary insertion order.",
         'def canonical_json_digest(value):\n    import hashlib\n    import json\n'
         '    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")\n'
         '    return hashlib.sha256(encoded).hexdigest()\n',
         [case([None], "74234e98afe7498fb5daf1f36ac2d78acc339464f950703b8c019892f982b90b"),
          case([{"b": 1, "a": 2}], "d3626ac30a87e6f7a6428233b3c68299976865fa5508e4267c5415c76af7a772"),
          case([[1, 2]], "49a64717d5d4cb19952e6eac2946415cf6879adacf9908e7d872332d32c6e684"),
          case(["é"], "abc3932c9ec58f042750738c49034e7ee67cf556d6be86832ac3dda38decb703")],
         ('sort_keys=True', 'sort_keys=False'), "security"),
]
