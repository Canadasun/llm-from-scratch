"""Budget policy and narrowly scoped Azure management requests."""
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

POLICY = Path(__file__).with_name("budget-policy.json")
RG = "llm-from-scratch-rg"
APP = "llm-serve"


def policy():
    value = json.loads(POLICY.read_text())
    expected = {"currency": "CAD", "monthly_cap": 10, "alert_actual": 5,
                "stop_actual": 8, "stop_forecast": 10, "resource_group": RG,
                "app": APP, "owner_only": True}
    if any(value.get(k) != v for k, v in expected.items()):
        raise ValueError("BLOCKED: budget policy must match the owner-approved C$10 plan")
    if os.environ.get("RG", RG) != RG or os.environ.get("APP", APP) != APP:
        raise ValueError("BLOCKED: Azure operations are restricted to the LLM app")
    return value


def az(*args):
    result = subprocess.run(["az", *args, "--only-show-errors", "-o", "json"],
                            capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError("Azure operation failed; inspect Azure operation status without exposing secrets")
    return json.loads(result.stdout) if result.stdout.strip() else None


class Azure:
    def __init__(self):
        policy()
        self.subscription = os.environ.get("SUBSCRIPTION") or az("account", "show")["id"]
        if not re.fullmatch(r"[0-9a-fA-F-]{36}", self.subscription):
            raise ValueError("invalid subscription ID")
        self.scope = f"/subscriptions/{self.subscription}/resourceGroups/{RG}"
        self.app = self.scope + f"/providers/Microsoft.App/containerApps/{APP}"

    def request(self, method, resource, version, body=None):
        if not resource.lower().startswith(self.scope.lower() + "/providers/"):
            raise ValueError("Azure request outside approved resource group")
        args = ["rest", "--method", method, "--url",
                "https://management.azure.com" + resource + "?api-version=" + version]
        if body is None:
            return az(*args)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "request.json"
            path.write_text(json.dumps(body))
            path.chmod(0o600)
            return az(*args, "--body", "@" + str(path))


def check_rollout():
    value = policy()
    if not value.get("go_live_approved"):
        raise ValueError("BLOCKED: Phase 7 owner go-live approval and qualification are still required")
    from deploy.lockdown import verify_controls
    verify_controls(Azure())


if __name__ == "__main__":
    try:
        check_rollout()
    except (ValueError, RuntimeError) as error:
        raise SystemExit(str(error))
