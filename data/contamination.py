"""Train/eval contamination guard.

A model that memorises its training data can look like it improved when it only
regurgitated the answer. The defence is to keep the evaluation OUT of the
training corpus and to verify that mechanically (external audit: split evaluation
so memorisation cannot masquerade as progress). This module checks that none of
the evaluation material — the held-out text and every task's prompt / target /
choices — appears verbatim in a training corpus.

It is a necessary, not sufficient, check: it catches exact overlap, not
paraphrase. Keep eval authored independently of the corpus as the primary
control; this makes a regression in that discipline fail loudly.
"""

import json
import os

# Only flag reasonably long verbatim overlaps, so incidental short strings
# (a keyword, a common identifier) are not false positives.
MIN_SNIPPET = 40


def _eval_strings(eval_dir):
    """Yield (source, text) for every evaluation string worth guarding."""
    heldout = os.path.join(eval_dir, "heldout_security.txt")
    if os.path.isfile(heldout):
        with open(heldout, encoding="utf-8") as fh:
            yield ("heldout_security.txt", fh.read())
    for name in sorted(os.listdir(eval_dir)):
        if not name.endswith(".jsonl"):
            continue
        path = os.path.join(eval_dir, name)
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    task = json.loads(line)
                except ValueError:
                    continue
                for field in ("prompt", "target"):
                    value = task.get(field)
                    if isinstance(value, str):
                        yield (f"{name}:{task.get('id', '?')}:{field}", value)
                for choice in task.get("choices", []) or []:
                    if isinstance(choice, str):
                        yield (f"{name}:{task.get('id', '?')}:choice", choice)


def find_contamination(corpus_text, eval_dir, min_snippet=MIN_SNIPPET):
    """Return a list of (source, snippet) for eval strings that appear verbatim in
    corpus_text. For long texts (the held-out file), a window of length
    min_snippet is scanned so partial overlap is caught too."""
    hits = []
    for source, text in _eval_strings(eval_dir):
        text = text.strip()
        if len(text) < min_snippet:
            continue
        if len(text) <= 300:
            if text in corpus_text:
                hits.append((source, text[:120]))
            continue
        # Long text: slide a window so a substantial excerpt is detected.
        step = min_snippet
        for i in range(0, len(text) - min_snippet, step):
            window = text[i:i + min_snippet]
            if window in corpus_text:
                hits.append((source, window))
                break
    return hits


def assert_clean(corpus_text, eval_dir, min_snippet=MIN_SNIPPET):
    """Raise ValueError if any evaluation string leaks into the corpus."""
    hits = find_contamination(corpus_text, eval_dir, min_snippet)
    if hits:
        preview = "; ".join(f"{s} -> {t!r}" for s, t in hits[:5])
        raise ValueError(
            f"train/eval contamination: {len(hits)} eval string(s) appear in the "
            f"corpus (memorisation risk). First: {preview}")
