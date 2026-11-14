import argparse
import hashlib
import json
from pathlib import Path


def compare(scores, base_sha256, candidate_sha256):
    modes = {}
    for mode in ("completion", "chat"):
        base, candidate = [scores[f"{label}-{mode}"] for label in ("base", "candidate")]
        for label, score, expected in (("base", base, base_sha256), ("candidate", candidate, candidate_sha256)):
            meta = score["meta"]
            if meta["checkpoint_sha256"] != expected or meta["decoding"]["chat"] != (mode == "chat"):
                raise ValueError(f"{label}-{mode}: checkpoint or decoding mode mismatch")
            if not meta["sandbox"]["available"] or any(row["unavailable"] for row in score["summary"].values()):
                raise ValueError("sandbox measurement unavailable")
            if any(row.get("status") in {"error", "unavailable"} for row in score["results"]):
                raise ValueError("functional measurement contains infrastructure errors")
            if meta["limit"] != 25 or not meta["decoding"]["greedy"] or meta["decoding"].get("max_new_tokens") != 384:
                raise ValueError("expected the declared 25-task greedy pilot")
            if set(score["summary"]) != {row["suite"] for row in score["results"]}:
                raise ValueError("summary and result suite membership differ")
            for suite, summary in score["summary"].items():
                rows = [row for row in score["results"] if row["suite"] == suite]
                if (len(rows) != summary["tasks"] or any(type(row["passed"]) is not bool for row in rows)
                        or sum(row["passed"] for row in rows) != summary["passed"]):
                    raise ValueError("summary does not match measured functional results")
        for field in ("bench_manifest_sha256", "behaviour_sha256", "suites", "suite_sha256", "decoding", "limit"):
            if base["meta"][field] != candidate["meta"][field]:
                raise ValueError(f"benchmark identity/configuration mismatch: {field}")
        index = lambda score: {(row["suite"], row["id"]): row for row in score["results"]}
        a, b = index(base), index(candidate)
        if a.keys() != b.keys() or len(a) != len(base["results"]) or len(b) != len(candidate["results"]):
            raise ValueError("functional task membership differs or contains duplicates")
        if base["summary"].keys() != candidate["summary"].keys():
            raise ValueError("suite membership differs")
        for suite in base["summary"]:
            if base["summary"][suite]["tasks"] != 25 or candidate["summary"][suite]["tasks"] != 25:
                raise ValueError("suite sample size differs from the declared pilot")
        if base["behaviour"].keys() != candidate["behaviour"].keys():
            raise ValueError("behavior category membership differs")
        for category in base["behaviour"]:
            if base["behaviour"][category]["n"] != candidate["behaviour"][category]["n"]:
                raise ValueError("behavior sample sizes differ")
        code_retained = all(candidate["summary"][s]["passed"] >= row["passed"] for s, row in base["summary"].items())
        code_improved = any(candidate["summary"][s]["passed"] > row["passed"] for s, row in base["summary"].items())
        syntax_retained = candidate["parses_as_python"] >= base["parses_as_python"]
        behavior_retained = all(candidate["behaviour"][c]["correct"] >= row["correct"] for c, row in base["behaviour"].items())
        modes[mode] = {
            "base": {key: base[key] for key in ("summary", "parses_as_python", "behaviour")},
            "candidate": {key: candidate[key] for key in ("summary", "parses_as_python", "behaviour")},
            "functional_gains": [list(key) for key in a if not a[key]["passed"] and b[key]["passed"]],
            "functional_losses": [list(key) for key in a if a[key]["passed"] and not b[key]["passed"]],
            "checks": {"code_retained": code_retained, "code_improved": code_improved,
                       "syntax_retained": syntax_retained, "behavior_retained": behavior_retained},
            "passed": code_retained and syntax_retained and behavior_retained and (code_improved or mode == "chat")}
    passed = all(row["passed"] for row in modes.values())
    return {"decision": "advance_to_full_qualification" if passed else "reject_candidate_keep_phase4_base",
            "passed": passed, "checkpoint_sha256": {"base": base_sha256, "candidate": candidate_sha256},
            "scope": "observed public 25-task-per-suite screen; not production qualification", "modes": modes}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Apply the predeclared matched-mode curriculum pilot screen.")
    parser.add_argument("--scores-dir", required=True)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    paths = {f"{label}-{mode}": Path(args.scores_dir) / f"scores-{label}-{mode}.json"
             for label in ("base", "candidate") for mode in ("completion", "chat")}
    result = compare({label: json.loads(path.read_text()) for label, path in paths.items()},
                     args.base_sha256, args.candidate_sha256)
    result["inputs_sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths.values()}
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"decision": result["decision"], "passed": result["passed"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
