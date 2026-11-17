"""Original public diagnostic tasks. Never import benchmark or private-holdout data."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Probe:
    family: str
    split: str
    name: str
    arguments: str
    spec: str
    body: str
    cases: list

    @property
    def header(self):
        return f"def {self.name}({self.arguments}):\n"


def c(args, expected):
    return {"args": args, "expected": expected}


TASKS = (
    Probe("consecutive-value-runs", "train", "value_runs", "items",
          "items is a list of integers. Return consecutive equal runs as [value, count] pairs, in order.",
          '    result = []\n    for item in items:\n        if result and result[-1][0] == item:\n'
          '            result[-1][1] += 1\n        else:\n            result.append([item, 1])\n    return result\n',
          [c([[]], []), c([[2, 2, 3, 2]], [[2, 2], [3, 1], [2, 1]]), c([[0, 0, 0]], [[0, 3]])]),
    Probe("cumulative-signed-totals", "train", "running_totals", "changes",
          "Return the cumulative totals of an integer list, starting from zero; do not include the initial zero.",
          '    total = 0\n    result = []\n    for change in changes:\n        total += change\n'
          '        result.append(total)\n    return result\n',
          [c([[]], []), c([[3, -5, 2]], [3, -2, 0]), c([[0, 4]], [0, 4])]),
    Probe("integer-session-lifetime", "train", "remaining_lifetime", "expires, now",
          "Both arguments are integer timestamps. Return expires minus now, clamped to a minimum of zero.",
          '    return max(0, expires - now)\n',
          [c([20, 7], 13), c([7, 7], 0), c([3, 8], 0), c([-2, -5], 3)]),
    Probe("adjacent-integer-differences", "train", "adjacent_changes", "values",
          "For a list of integers, return each value minus its immediately preceding value, in order.",
          '    return [values[i] - values[i - 1] for i in range(1, len(values))]\n',
          [c([[]], []), c([[5]], []), c([[4, 1, 7, 7]], [-3, 6, 0])]),
    Probe("two-channel-event-tally", "train", "tally_outcomes", "events",
          "events contains strings. Return [allowed_count, denied_count], counting exact 'allow' and 'deny' only.",
          '    return [events.count("allow"), events.count("deny")]\n',
          [c([[]], [0, 0]), c([["deny", "allow", "ALLOW", "deny"]], [1, 2]), c([["other"]], [0, 0])]),
    Probe("bounded-numeric-distance", "train", "within_distance", "left, right, radius",
          "Arguments are integers. Return False for negative radius; otherwise whether abs(left-right) <= radius.",
          '    return radius >= 0 and abs(left - right) <= radius\n',
          [c([1, 4, 3], True), c([1, 4, 2], False), c([0, 0, -1], False), c([-4, -2, 2], True)]),
    Probe("first-strict-threshold-position", "evaluation", "first_above", "values, threshold",
          "Return the index of the first integer in values strictly greater than threshold; return -1 if absent.",
          '    for index, value in enumerate(values):\n        if value > threshold:\n            return index\n'
          '    return -1\n',
          [c([[], 2], -1), c([[2, 2, 3], 2], 2), c([[5, 6], 4], 0), c([[-2, -1], 0], -1)]),
    Probe("alternating-sequence-merge", "evaluation", "interleave", "left, right",
          "Return a new list alternating left then right elements; append remaining elements of the longer list.",
          '    result = []\n    for index in range(max(len(left), len(right))):\n'
          '        if index < len(left):\n            result.append(left[index])\n'
          '        if index < len(right):\n            result.append(right[index])\n    return result\n',
          [c([[], []], []), c([[1, 2], [8]], [1, 8, 2]), c([[], [3, 4]], [3, 4]),
           c([[1], [2, 3]], [1, 2, 3])]),
    Probe("literal-pair-reversal", "evaluation", "reverse_pairs", "pairs",
          "pairs is a list of two-element lists. Return a new list with the two elements of each pair swapped.",
          '    return [[pair[1], pair[0]] for pair in pairs]\n',
          [c([[]], []), c([[["a", 2], [None, False]]], [[2, "a"], [False, None]]), c([[[1, 1]]], [[1, 1]])]),
    Probe("nondecreasing-order-check", "evaluation", "is_nondecreasing", "values",
          "Return whether an integer list is nondecreasing; the empty list and singleton lists return True.",
          '    return all(values[i - 1] <= values[i] for i in range(1, len(values)))\n',
          [c([[]], True), c([[3]], True), c([[1, 1, 2]], True), c([[2, 1]], False)]),
    Probe("aligned-request-budget", "evaluation", "whole_requests", "remaining, cost",
          "Arguments are integers. Return zero if remaining < 0 or cost <= 0, else remaining // cost.",
          '    if remaining < 0 or cost <= 0:\n        return 0\n    return remaining // cost\n',
          [c([11, 3], 3), c([2, 3], 0), c([-1, 2], 0), c([10, 0], 0), c([0, -2], 0)]),
    Probe("exact-required-scope-subset", "evaluation", "has_scopes", "granted, required",
          "Both lists contain strings. Return whether every required value occurs exactly (case-sensitive) in granted.",
          '    return all(scope in granted for scope in required)\n',
          [c([[], []], True), c([["read"], ["read", "read"]], True), c([["Read"], ["read"]], False),
           c([["read"], ["write"]], False)]),
)
