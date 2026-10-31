"""Author-generated seed instruction data; never reads benchmark task files.

Examples are deliberately small, original templates, not a claim of a complete
secure-coding curriculum. Family-level splits prevent renamed copies of the
same exercise from appearing in both training and validation.
"""

import argparse
import hashlib
import json
from pathlib import Path


CODE = {
    "square": ("value", "return value * value", "returns the square of a number"),
    "absolute": ("value", "return -value if value < 0 else value", "returns a number's absolute value"),
    "clip": ("value, low, high", "return max(low, min(high, value))", "clips value to the inclusive range low to high"),
    "nonempty": ("values", "return [v for v in values if v]", "removes falsy items from a list"),
    "trim": ("text", "return text.strip()", "removes leading and trailing whitespace"),
    "normalize": ("text", "return text.strip().lower()", "trims whitespace and lowercases text"),
    "positive": ("values", "return sum(v for v in values if v > 0)", "sums only positive numbers in a list"),
    "prefix": ("text, prefix", "return text.startswith(prefix)", "checks whether text starts with prefix"),
    "suffix": ("text, suffix", "return text.endswith(suffix)", "checks whether text ends with suffix"),
    "default": ("value, fallback", "return fallback if value is None else value", "uses fallback only when value is None"),
    "lengths": ("words", "return [len(word) for word in words]", "returns a list of word lengths"),
    "lookup": ("mapping, key, fallback", "return mapping.get(key, fallback)", "looks up key with a fallback for missing keys"),
    "count_lines": ("text", "return len(text.splitlines())", "counts lines using Python splitlines semantics"),
    "log_line": ("text", "return text.replace('\\r', ' ').replace('\\n', ' ')", "replaces carriage returns and newlines with spaces"),
    "valid_port": ("value", "return type(value) is int and 1 <= value <= 65535", "checks for an integer TCP port from 1 to 65535, rejecting booleans"),
    "mask": ("text", "return '*' * max(0, len(text) - 4) + text[-4:] if len(text) > 4 else '*' * len(text)", "masks all but the final four characters, masking everything for shorter strings"),
}

QA = {
    "authentication": ("Distinguish authentication from authorization.", "Authentication establishes identity. Authorization checks whether that identity may perform a particular action."),
    "permissions": ("What does least privilege require for a service account?", "Grant only the permissions and resources needed for its job. Review access regularly and revoke permissions when no longer needed."),
    "transactions": ("Why should related database writes use a transaction?", "A transaction commits the related writes together or rolls them back together, avoiding partially applied state."),
    "backups": ("How can a team check that its backups are useful?", "Perform restore tests in an isolated environment and verify the restored data and recovery procedure, not just the presence of backup files."),
    "logging": ("What information should be kept out of routine application logs?", "Passwords, access tokens, private keys and unnecessary personal data should not be logged. Use controlled redaction and limited retention."),
    "review": ("Does a passing unit test prove that a program has no vulnerabilities?", "No. Tests cover specific cases. Combine tests with code review, threat modeling and security analysis, and state the checks' limitations."),
    "dependencies": ("How should a project manage third-party dependencies?", "Track dependency versions and provenance, review updates and advisories, and test changes before deployment."),
    "incident": ("What should an incident timeline record?", "Record timestamps, observations, affected assets and actions taken. Preserve evidence and distinguish confirmed facts from hypotheses."),
    "secrets": ("Where should a production application obtain its credentials?", "Use an approved secret-management mechanism with scoped access and rotation. Do not embed live credentials in source code."),
    "input": ("What is useful input validation at a trust boundary?", "Validate the input's type, size, range and allowed structure before expensive or privileged processing. Validation complements downstream controls."),
    "uncertainty": ("What should an assistant do when evidence is insufficient?", "State what is known, explain the uncertainty and request the missing evidence. Do not invent facts or claim a check was performed."),
    "verification": ("How should a proposed code change be verified?", "Review the diff, run relevant tests and security checks, and record the results and limitations before approving the change."),
}


def examples():
    rows = []
    for family, (arguments, body, description) in CODE.items():
        for index in range(20):
            name = f"{family}_helper_{index}"
            prompt = (f"Implement a Python function named {name}({arguments}) that {description}.\n\n```python\n"
                      if index % 2 else f"Write Python code for {name}({arguments}) which {description}. Return only the function.\n")
            response = f"def {name}({arguments}):\n    {body}\n"
            rows.append({"id": f"code-{family}-{index}", "family": f"code-{family}",
                         "prompt": prompt, "response": response})
    for family, (question, answer) in QA.items():
        for index, prefix in enumerate(("", "Explain briefly: ", "For a developer: ", "For a security review: ")):
            rows.append({"id": f"qa-{family}-{index}", "family": f"qa-{family}",
                         "prompt": f"Q: {prefix}{question}\nA: ", "response": answer + "\n"})
    return rows


def split_rows(rows):
    validation = {"code-clip", "code-prefix", "code-lengths", "qa-backups", "qa-incident"}
    return ([r for r in rows if r["family"] not in validation],
            [r for r in rows if r["family"] in validation])


def build(directory):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    train, validation = split_rows(examples())
    hashes = {}
    for name, rows in (("train", train), ("validation", validation)):
        data = ''.join(json.dumps(r, sort_keys=True) + '\n' for r in rows)
        (root / f"{name}.jsonl").write_text(data)
        hashes[name] = hashlib.sha256(data.encode()).hexdigest()
    manifest = {"source": "original templates in data/build_instructions.py; no evaluation files read",
                "license": "MIT (project-authored)", "train_examples": len(train),
                "validation_examples": len(validation), "split": "disjoint exercise families",
                "sha256": hashes}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/instructions")
    print(json.dumps(build(parser.parse_args().output), indent=2))
