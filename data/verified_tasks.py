from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    family: str
    split: str
    spec: str
    source: str
    cases: list
    mutation: tuple
    domain: str = "python"
    source_file: str = "data/verified_tasks.py"


def case(args, expected=None, raises=None):
    return {"args": args, **({"raises": raises} if raises else {"expected": expected})}


TASKS = [
    Task("comma-header-values", "train",
         "Implement header_values(text). Split a comma-separated HTTP-style value list, strip whitespace, "
         "lowercase each value, discard empty entries, and keep duplicates in their original order.",
         'def header_values(text):\n    return [part.strip().lower() for part in text.split(",") if part.strip()]\n',
         [case([""], []), case([" GZIP, br ,,GZIP "], ["gzip", "br", "gzip"]),
          case([", ,"], []), case(["TEXT/HTML"], ["text/html"])], (".lower()", "")),
    Task("mapping-secret-redaction", "train",
         "Implement redact_fields(record, sensitive). Return a new dictionary preserving keys and values except "
         "keys whose lowercase form is in sensitive (compared case-insensitively): replace those values with '***'. "
         "Do not modify the input dictionary.",
         'def redact_fields(record, sensitive):\n    hidden = {key.lower() for key in sensitive}\n'
         '    return {key: "***" if key.lower() in hidden else value for key, value in record.items()}\n',
         [case([{}, []], {}), case([{"Token": "abc", "x": 2}, ["TOKEN"]], {"Token": "***", "x": 2}),
          case([{"password": None}, ["password"]], {"password": "***"}),
          case([{"token_hint": "ok"}, ["token"]], {"token_hint": "ok"})], ("key.lower() in hidden", "key in hidden")),
    Task("metric-label-escaping", "train",
         "Implement escape_label(text) for a metric label value. In that order, escape each backslash as two "
         "backslashes, each newline as backslash plus n, and each double quote as backslash plus double quote. "
         "Leave other characters unchanged.",
         'def escape_label(text):\n    return text.replace("\\\\", "\\\\\\\\").replace("\\n", "\\\\n").replace(\'"\', \'\\\\"\')\n',
         [case([""], ""), case(['a"b'], 'a\\"b'), case(["a\nb"], "a\\nb"),
          case(["a\\b"], "a\\\\b")], ('.replace("\\n", "\\\\n")', "")),
    Task("log-key-value-pairs", "train",
         "Implement parse_log_pairs(text). Split on whitespace, ignore tokens without '=', split each remaining "
         "token at its first '=', ignore empty keys, and return a dict. Retain empty values and let the last "
         "occurrence of a key win. Do not interpret quotes or escapes.",
         'def parse_log_pairs(text):\n    result = {}\n    for token in text.split():\n'
         '        key, separator, value = token.partition("=")\n        if separator and key:\n'
         '            result[key] = value\n    return result\n',
         [case([""], {}), case(["x=1 bare y=a=b"], {"x": "1", "y": "a=b"}),
          case(["x=1 x=2"], {"x": "2"}), case(["=bad empty="], {"empty": ""})],
         ('if separator and key:', 'if separator:')),
    Task("pagination-clamping", "train",
         "Implement clamp_page_size(requested, maximum). Both arguments are integers. Raise ValueError if maximum "
         "is less than 1; otherwise clamp requested to the inclusive range 1..maximum.",
         'def clamp_page_size(requested, maximum):\n    if maximum < 1:\n        raise ValueError("maximum")\n'
         '    return min(maximum, max(1, requested))\n',
         [case([0, 10], 1), case([100, 10], 10), case([4, 10], 4),
          case([-5, 3], 1), case([2, 0], raises="ValueError")], ('max(1, requested)', 'requested')),
    Task("literal-env-entry", "train",
         "Implement parse_env_entry(line). Strip the line. Return None for an empty line, a line starting with '#', "
         "a line without '=', or a blank key. Otherwise split at the first '=', strip the key and value, "
         "and return [key, value]. Preserve quotes, dollar signs and embedded '=' literally; no expansion.",
         'def parse_env_entry(line):\n    line = line.strip()\n    if not line or line.startswith("#"):\n'
         '        return None\n    key, separator, value = line.partition("=")\n'
         '    if not separator or not key.strip():\n        return None\n'
         '    return [key.strip(), value.strip()]\n',
         [case([" # x=y"], None), case([" A = x=y "], ["A", "x=y"]),
          case(["A='$HOME'"], ["A", "'$HOME'"]), case(["=x"], None),
          case(["no-entry"], None), case(["A="], ["A", ""])], ('value.strip()', 'value')),
    Task("none-skipping-overlay", "train",
         "Implement overlay_config(base, override). Return a new dictionary with base's entries and override's "
         "non-None entries replacing them. None overrides mean keep the base entry (or omit if absent). "
         "Values such as False, 0 and the empty string must still override. Do not mutate either input.",
         'def overlay_config(base, override):\n    result = dict(base)\n'
         '    result.update({key: value for key, value in override.items() if value is not None})\n'
         '    return result\n',
         [case([{}, {}], {}), case([{"a": 1}, {"a": None, "b": None}], {"a": 1}),
          case([{"a": 1}, {"a": 0}], {"a": 0}), case([{}, {"x": False, "y": ""}], {"x": False, "y": ""})],
         ('value is not None', 'value')),
    Task("token-suffix-masking", "train",
         "Implement mask_token(token, visible). visible is an integer and must be nonnegative, else raise ValueError. "
         "If visible is 0 or the token length is at most visible, replace the whole token with '*'. "
         "Otherwise preserve the final visible characters and replace the preceding characters with '*'.",
         'def mask_token(token, visible):\n    if visible < 0:\n        raise ValueError("visible")\n'
         '    if visible == 0 or len(token) <= visible:\n        return "*" * len(token)\n'
         '    return "*" * (len(token) - visible) + token[-visible:]\n',
         [case(["abcdef", 2], "****ef"), case(["abc", 3], "***"),
          case(["abc", 0], "***"), case(["", 2], ""), case(["x", -1], raises="ValueError")],
         ('len(token) <= visible', 'len(token) < visible')),
    Task("bounded-content-length", "train",
         "Implement parse_content_length(text). Strip surrounding whitespace; accept only 1 to 10 ASCII digits "
         "representing a value no greater than 2147483647. Return the integer or raise ValueError. "
         "Reject signs, non-ASCII digits, commas and internal whitespace.",
         'def parse_content_length(text):\n    text = text.strip()\n'
         '    if not 1 <= len(text) <= 10 or any(c not in "0123456789" for c in text):\n'
         '        raise ValueError("length")\n    value = int(text)\n'
         '    if value > 2147483647:\n        raise ValueError("length")\n    return value\n',
         [case([" 12 "], 12), case(["0"], 0), case(["+2"], raises="ValueError"),
          case(["٢"], raises="ValueError"), case(["2147483648"], raises="ValueError"),
          case([""], raises="ValueError"), case(["1,2"], raises="ValueError")],
         ('value > 2147483647', 'value > 9999999999')),
    Task("ordered-header-lookup", "train",
         "Implement first_header(pairs, name). pairs is a list of [key, value] string pairs. Return the value of "
         "the first key equal to name ignoring case, or None when absent. Do not strip or otherwise alter values.",
         'def first_header(pairs, name):\n    for key, value in pairs:\n'
         '        if key.lower() == name.lower():\n            return value\n    return None\n',
         [case([[], "x"], None), case([[["X", " a "]], "x"], " a "),
          case([[["X", "a"], ["x", "b"]], "x"], "a"), case([[["xx", "a"]], "x"], None)],
         ('key.lower() == name.lower()', 'key == name')),
    Task("literal-boolean-setting", "train",
         "Implement parse_bool_setting(text). Strip whitespace and ignore case. Accept true, yes, on, 1 as True "
         "and false, no, off, 0 as False. Reject every other value with ValueError.",
         'def parse_bool_setting(text):\n    value = text.strip().lower()\n'
         '    if value in {"true", "yes", "on", "1"}:\n        return True\n'
         '    if value in {"false", "no", "off", "0"}:\n        return False\n'
         '    raise ValueError("boolean")\n',
         [case([" YES "], True), case(["Off"], False), case(["1"], True),
          case(["0"], False), case(["maybe"], raises="ValueError"), case([""], raises="ValueError")],
         ('"false", "no", "off", "0"', '"false", "no", "off"')),
    Task("ascii-log-sanitization", "train",
         "Implement clean_log_text(text). Remove ASCII control characters with code points 0 through 31 and 127. "
         "Preserve spaces and every other character, including non-ASCII text.",
         'def clean_log_text(text):\n    return "".join(c for c in text if ord(c) >= 32 and ord(c) != 127)\n',
         [case(["a\nb\tc"], "abc"), case([" a "], " a "), case(["café\x7f"], "café"),
          case(["\x00\x1f"], ""), case([""], "")], (' and ord(c) != 127', '')),
    Task("suffix-media-lookup", "train",
         "Implement media_type(filename, types, fallback). Find the text after the final '.' in the final '/' "
         "path component, lowercase it, and look it up in types. Return fallback if there is no dot or no mapping. "
         "Do not treat dots in directory names as a file extension; types has lowercase keys without dots.",
         'def media_type(filename, types, fallback):\n    leaf = filename.rsplit("/", 1)[-1]\n'
         '    if "." not in leaf:\n        return fallback\n'
         '    return types.get(leaf.rsplit(".", 1)[-1].lower(), fallback)\n',
         [case(["a.JSON", {"json": "application/json"}, "unknown"], "application/json"),
          case(["a.json/readme", {"json": "j"}, "unknown"], "unknown"),
          case(["a.tar.gz", {"gz": "gzip"}, "?"], "gzip"), case(["file", {}, "?"], "?")],
         ('.lower()', '')),
    Task("query-key-whitelisting", "train",
         "Implement filter_query(pairs, allowed). Return a new list of [key, value] pairs whose key is in allowed. "
         "Keep original order and duplicates; compare keys case-sensitively.",
         'def filter_query(pairs, allowed):\n    return [[key, value] for key, value in pairs if key in allowed]\n',
         [case([[], ["q"]], []), case([[["q", "1"], ["x", "2"], ["q", "3"]], ["q"]], [["q", "1"], ["q", "3"]]),
          case([[["Q", "1"]], ["q"]], []), case([[["q", ""]], []], [])],
         ('if key in allowed', 'if key not in allowed')),
    Task("status-class-counter", "train",
         "Implement count_status_classes(codes). codes contains integers. Count only status codes from 100 through "
         "599 inclusive into a dict whose keys are '1xx' through '5xx'. Omit classes with zero count.",
         'def count_status_classes(codes):\n    result = {}\n    for code in codes:\n'
         '        if 100 <= code <= 599:\n            key = str(code // 100) + "xx"\n'
         '            result[key] = result.get(key, 0) + 1\n    return result\n',
         [case([[]], {}), case([[200, 201, 404, 599]], {"2xx": 2, "4xx": 1, "5xx": 1}),
          case([[99, 600, -1]], {}), case([[100, 599]], {"1xx": 1, "5xx": 1})],
         ('100 <= code <= 599', '100 < code < 599')),
    Task("case-insensitive-prefix-drop", "train",
         "Implement strip_bearer(value). Remove exactly the first seven characters when value begins with "
         "'Bearer ' ignoring case, then strip the remainder. Otherwise return None. Do not accept 'Bearer' "
         "without the space. An empty remainder is the empty string.",
         'def strip_bearer(value):\n    if value[:7].lower() != "bearer ":\n        return None\n'
         '    return value[7:].strip()\n',
         [case(["Bearer abc"], "abc"), case(["bEaReR  x "], "x"), case(["Bearer"], None),
          case(["Basic abc"], None), case(["Bearer "], "")],
         ('value[:7].lower()', 'value[:7]')),
    Task("copy-rename-field", "train",
         "Implement rename_field(record, old, new). Return a copy. If old exists and differs from new, move its "
         "value to new, replacing any existing new value. If old is absent or old equals new, leave entries alone. "
         "Do not mutate record.",
         'def rename_field(record, old, new):\n    result = dict(record)\n'
         '    if old in result and old != new:\n        result[new] = result.pop(old)\n    return result\n',
         [case([{"a": 1}, "a", "b"], {"b": 1}), case([{"a": 1}, "a", "a"], {"a": 1}),
          case([{"a": None, "b": 2}, "a", "b"], {"b": None}), case([{}, "a", "b"], {})],
         ('result.pop(old)', 'result[old]')),
    Task("relative-import-depth", "train",
         "Implement relative_import_depth(name). Count consecutive '.' characters at the start of name, "
         "stopping at the first different character. Dots elsewhere do not count.",
         'def relative_import_depth(name):\n    return len(name) - len(name.lstrip("."))\n',
         [case(["...pkg.mod"], 3), case(["pkg.mod"], 0), case([""], 0), case([".."], 2)],
         ('name.lstrip(".")', 'name.replace(".", "")')),
    Task("retry-delay-cap", "validation",
         "Implement retry_delay(attempt, base, cap). The three values are nonnegative integers. "
         "Raise ValueError if any is negative. Return min(cap, base * 2**attempt).",
         'def retry_delay(attempt, base, cap):\n    if min(attempt, base, cap) < 0:\n        raise ValueError("negative")\n'
         '    return min(cap, base * 2 ** attempt)\n',
         [case([0, 3, 10], 3), case([2, 3, 10], 10), case([1, 0, 5], 0),
          case([-1, 3, 10], raises="ValueError")], ('min(cap, base * 2 ** attempt)', 'base * 2 ** attempt')),
    Task("indentation-measure", "validation",
         "Implement indentation_width(line, tab_width). tab_width is a positive integer; reject other integers "
         "with ValueError. Count the leading spaces and tabs only. Each space counts 1; each tab counts tab_width "
         "(not tab stops). Stop at the first other character.",
         'def indentation_width(line, tab_width):\n    if tab_width <= 0:\n        raise ValueError("tab width")\n'
         '    total = 0\n    for c in line:\n        if c == " ":\n            total += 1\n'
         '        elif c == "\\t":\n            total += tab_width\n        else:\n            break\n    return total\n',
         [case([" \tx", 4], 5), case(["x  ", 4], 0), case(["", 2], 0),
          case(["  ", 0], raises="ValueError"), case(["\t\t", 3], 6)], ('total += tab_width', 'total += 1')),
    Task("config-section-diff", "validation",
         "Implement changed_keys(before, after). Return the sorted list of keys added, removed or assigned "
         "unequal values between the dictionaries. All keys are strings. A missing key differs from a key "
         "whose value is None.",
         'def changed_keys(before, after):\n    return sorted(key for key in before.keys() | after.keys()\n'
         '                  if key not in before or key not in after or before[key] != after[key])\n',
         [case([{}, {}], []), case([{}, {"x": None}], ["x"]), case([{"b": 1, "a": 2}, {"a": 3}], ["a", "b"]),
          case([{"x": 2}, {"x": 2}], [])], ('before.keys() | after.keys()', 'before.keys() & after.keys()')),
    Task("line-ending-normalization", "validation",
         "Implement normalize_line_endings(text). Replace CRLF with LF, then every remaining CR with LF. "
         "Preserve all other characters and the number of logical line breaks; do not add a trailing newline.",
         'def normalize_line_endings(text):\n    return text.replace("\\r\\n", "\\n").replace("\\r", "\\n")\n',
         [case(["a\r\nb\rc\n"], "a\nb\nc\n"), case(["abc"], "abc"), case(["\r\n"], "\n"),
          case([""], ""), case(["\r\r"], "\n\n")], ('.replace("\\r", "\\n")', '')),
    Task("posix-mode-write-bit", "validation",
         "Implement is_world_writable(mode). mode is a nonnegative integer representing Unix permission bits. "
         "Return True exactly when bit 0o002 is set; do not treat group-write as world-write.",
         'def is_world_writable(mode):\n    return bool(mode & 0o002)\n',
         [case([0o644], False), case([0o666], True), case([0o620], False), case([0], False), case([0o100002], True)],
         ('0o002', '0o020')),
    Task("duration-unit-parser", "validation",
         "Implement parse_duration(text). Accept an ASCII digit string immediately followed by 'ms', 's' or 'm', "
         "after stripping outer whitespace. Return integer milliseconds. Reject uppercase units, missing digits, "
         "signs and internal spaces with ValueError.",
         'def parse_duration(text):\n    text = text.strip()\n    for suffix, factor in (("ms", 1), ("s", 1000), ("m", 60000)):\n'
         '        if text.endswith(suffix):\n            number = text[:-len(suffix)]\n'
         '            if number and all(c in "0123456789" for c in number):\n                return int(number) * factor\n'
         '    raise ValueError("duration")\n',
         [case(["12ms"], 12), case([" 2s "], 2000), case(["1m"], 60000),
          case(["-1s"], raises="ValueError"), case(["1 s"], raises="ValueError"), case(["ms"], raises="ValueError")],
         ('("m", 60000)', '("m", 1000)')),
    Task("csv-formula-prefix", "test",
         "Implement escape_csv_cell(text). Return an apostrophe followed by the original text if its first "
         "non-space/tab character is any of =, +, -, @. Otherwise return text unchanged. This is a literal "
         "cell transformation, not a complete CSV writer; preserve original whitespace.",
         'def escape_csv_cell(text):\n    stripped = text.lstrip(" \\t")\n'
         '    if stripped and stripped[0] in "=+-@":\n        return "\'" + text\n    return text\n',
         [case(["=SUM(A1)"], "'=SUM(A1)"), case([" \t+1"], "' \t+1"), case(["ordinary"], "ordinary"),
          case([""], ""), case(["   "], "   "), case(["@x"], "'@x")],
         ('text.lstrip(" \\t")', 'text')),
    Task("etag-weak-prefix", "test",
         "Implement etag_payload(text). Strip outer whitespace, remove an optional exact uppercase 'W/' prefix, "
         "then accept only a pair of surrounding double quotes with no internal double quote, CR or LF. "
         "Return the enclosed string or None. Do not strip whitespace after removing W/.",
         'def etag_payload(text):\n    text = text.strip()\n    if text.startswith("W/"):\n        text = text[2:]\n'
         '    if len(text) < 2 or text[0] != \'"\' or text[-1] != \'"\':\n        return None\n'
         '    value = text[1:-1]\n    if any(c in value for c in (\'"\', "\\r", "\\n")):\n        return None\n'
         '    return value\n',
         [case([' W/"abc" '], "abc"), case(['""'], ""), case(['W/ "x"'], None),
          case(['"a"b"'], None), case(['"a\nb"'], None), case(["x"], None)],
         ('text = text[2:]', 'text = text[2:].strip()')),
    Task("numeric-version-order", "test",
         "Implement version_at_least(actual, minimum). Both are nonempty lists of nonnegative integers. "
         "Compare lexicographically after padding the shorter list with zeros to the longer length; "
         "return whether actual is at least minimum. Do not compare string representations.",
         'def version_at_least(actual, minimum):\n    width = max(len(actual), len(minimum))\n'
         '    left = actual + [0] * (width - len(actual))\n    right = minimum + [0] * (width - len(minimum))\n'
         '    return left >= right\n',
         [case([[1, 2], [1, 2, 0]], True), case([[1, 9], [1, 10]], False),
          case([[2], [1, 99]], True), case([[1], [1, 0, 1]], False)], ('return left >= right', 'return actual >= minimum')),
    Task("bounded-tail-lines", "test",
         "Implement tail_lines(text, limit). limit is an integer; raise ValueError if negative. Return the final "
         "limit lines using str.splitlines(), as a list. A zero limit must return an empty list.",
         'def tail_lines(text, limit):\n    if limit < 0:\n        raise ValueError("limit")\n'
         '    return text.splitlines()[-limit:] if limit else []\n',
         [case(["a\nb\nc\n", 2], ["b", "c"]), case(["a", 0], []), case(["", 3], []),
          case(["a\r\nb", 5], ["a", "b"]), case(["a", -1], raises="ValueError")],
         (' if limit else []', '')),
    Task("digest-name-canonicalization", "test",
         "Implement canonical_digest(name). Strip whitespace, lowercase, and remove '-' characters. "
         "Return the normalized name if it is sha256, sha384 or sha512; otherwise raise ValueError. "
         "Do not remove underscores or accept weak digest names.",
         'def canonical_digest(name):\n    normalized = name.strip().lower().replace("-", "")\n'
         '    if normalized not in {"sha256", "sha384", "sha512"}:\n        raise ValueError("digest")\n'
         '    return normalized\n',
         [case([" SHA-256 "], "sha256"), case(["sha512"], "sha512"), case(["sha_256"], raises="ValueError"),
          case(["md5"], raises="ValueError"), case(["sha1"], raises="ValueError")],
         ('"sha256", "sha384", "sha512"', '"sha256", "sha384", "sha512", "md5"')),
    Task("ordered-path-segments", "test",
         "Implement split_route(path). Discard the part starting at the first '?', then split on '/'. "
         "Discard empty segments but preserve '.' and '..' literally. Do not URL-decode or normalize segments.",
         'def split_route(path):\n    return [segment for segment in path.split("?", 1)[0].split("/") if segment]\n',
         [case(["/a//b/?q=/x"], ["a", "b"]), case(["/"], []), case(["../a%2Fb"], ["..", "a%2Fb"]),
          case(["?q=x"], []), case(["a/b"], ["a", "b"])],
         ('path.split("?", 1)[0]', 'path')),
]
