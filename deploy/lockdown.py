"""Owner-only ingress and budget stop controls for the existing LLM app."""
import argparse
from datetime import datetime, timezone
import ipaddress
import json
import os
import re
import time
import urllib.request
import uuid

from deploy.azure_guard import APP, RG, Azure, policy

APP_API = "2025-07-01"
LOGIC_API = "2019-05-01"
BUDGET_API = "2024-08-01"
WORKFLOW = "llm-budget-stopper"
BUDGET = "llm-monthly-10cad"
ACTION_GROUP = "llm-budget-stop"


def allow_ip(client, address=None):
    if address is None:
        with urllib.request.urlopen("https://api.ipify.org", timeout=15) as response:
            address = response.read(64).decode().strip()
    address = ipaddress.IPv4Address(address)
    app = client.request("GET", client.app, APP_API)
    rules = [{"name": "owner", "action": "Allow", "ipAddressRange": f"{address}/32",
              "description": "Owner access only"}]
    client.request("PATCH", client.app, APP_API,
                   {"location": app["location"], "properties": {"configuration": {"ingress": {"ipSecurityRestrictions": rules}}}})
    actual = client.request("GET", client.app, APP_API)["properties"]["configuration"]["ingress"]
    if actual["ipSecurityRestrictions"] != rules:
        raise ValueError("owner ingress readback failed")
    return {"owner_rule_count": 1, "owner_prefix_length": 32}


def workflow_body(app, location):
    return {"location": location, "identity": {"type": "SystemAssigned"}, "properties": {
        "state": "Enabled", "definition": {
            "$schema": "https://schema.management.azure.com/providers/Microsoft.Logic/schemas/2016-06-01/workflowdefinition.json#",
            "contentVersion": "1.0.0.0", "parameters": {},
            "triggers": {"budget": {"type": "Request", "kind": "Http",
                                     "inputs": {"schema": {"type": "object"}}}},
            "actions": {"stop_llm": {"type": "Http", "runAfter": {},
                "inputs": {"method": "POST", "uri": "https://management.azure.com" + app +
                    "/stop?api-version=" + APP_API,
                    "authentication": {"type": "ManagedServiceIdentity", "audience": "https://management.azure.com/"}},
                "runtimeConfiguration": {"secureData": {"properties": ["inputs", "outputs"]}}}},
            "outputs": {}}}}


def budget_body(group, email, now=None):
    now = now or datetime.now(timezone.utc)
    notifications = {}
    for name, threshold, kind, groups in (("actual_alert", 50, "Actual", []),
                                          ("actual_stop", 80, "Actual", [group]),
                                          ("forecast_stop", 100, "Forecasted", [group])):
        notifications[name] = {"enabled": True, "operator": "GreaterThanOrEqualTo",
                               "threshold": threshold, "thresholdType": kind,
                               "contactEmails": [email], "contactGroups": groups}
    return {"properties": {"category": "Cost", "amount": 10, "timeGrain": "Monthly",
        "timePeriod": {"startDate": now.strftime("%Y-%m-01T00:00:00Z"),
                       "endDate": now.replace(year=now.year + 2, day=1).strftime("%Y-%m-01T00:00:00Z")},
        "notifications": notifications}}


def setup_stopper(client):
    app = client.request("GET", client.app, APP_API)
    workflow = client.scope + "/providers/Microsoft.Logic/workflows/" + WORKFLOW
    budget = client.scope + "/providers/Microsoft.Consumption/budgets/" + BUDGET
    group = client.scope + "/providers/Microsoft.Insights/actionGroups/" + ACTION_GROUP
    existing = client.request("GET", client.scope + "/providers/Microsoft.Consumption/budgets", BUDGET_API)
    email = os.environ.get("ALERT_EMAIL")
    if not email:
        for item in existing["value"]:
            for notification in item["properties"].get("notifications", {}).values():
                emails = notification.get("contactEmails", [])
                if emails:
                    email = emails[0]
                    break
    if not email or "@" not in email or any(c in email for c in "\r\n"):
        raise ValueError("ALERT_EMAIL is required; no existing budget recipient found")
    if not existing["value"]:
        raise ValueError("an existing budget with verified CAD billing currency is required")
    for item in existing["value"]:
        spend = item["properties"].get("currentSpend", {})
        if spend.get("unit") != "CAD":
            raise ValueError("budget currency must be verified as CAD")
        if float(spend.get("amount", 0)) >= policy()["stop_actual"]:
            raise ValueError("stop: current spend has reached the C$8 threshold")
    result = client.request("PUT", workflow, LOGIC_API, workflow_body(client.app, app["location"]))
    principal = result["identity"]["principalId"]
    role_id = str(uuid.uuid5(uuid.NAMESPACE_URL, client.scope + "/llm-budget-stop-only"))
    role = client.scope + "/providers/Microsoft.Authorization/roleDefinitions/" + role_id
    role_result = client.request("PUT", role, "2022-04-01", {"properties": {
        "roleName": "LLM budget stop only", "description": "Stop the owner LLM app only",
        "type": "CustomRole", "assignableScopes": [client.scope],
        "permissions": [{"actions": ["Microsoft.App/containerApps/read", "Microsoft.App/containerApps/stop/action"],
                         "notActions": [], "dataActions": [], "notDataActions": []}]}})
    assignment = client.app + "/providers/Microsoft.Authorization/roleAssignments/" + str(
        uuid.uuid5(uuid.NAMESPACE_URL, client.app + principal + role_id))
    client.request("PUT", assignment, "2022-04-01", {"properties": {
        "roleDefinitionId": role_result["id"], "principalId": principal, "principalType": "ServicePrincipal"}})
    callback = client.request("POST", workflow + "/triggers/budget/listCallbackUrl", "2016-06-01")["value"]
    client.request("PUT", group, "2023-01-01", {"location": "Global", "properties": {
        "groupShortName": "llm-stop", "enabled": True, "logicAppReceivers": [{
            "name": "stop-llm", "resourceId": workflow, "callbackUrl": callback, "useCommonAlertSchema": False}]}})
    client.request("PUT", budget, BUDGET_API, budget_body(group, email))
    return verify_controls(client)


def verify_controls(client):
    app = client.request("GET", client.app, APP_API)
    ingress = app["properties"]["configuration"]["ingress"]
    rules = ingress.get("ipSecurityRestrictions") or []
    if len(rules) != 1 or rules[0]["action"] != "Allow":
        raise ValueError("exactly one owner ingress allow rule required")
    network = ipaddress.ip_network(rules[0]["ipAddressRange"])
    if network.version != 4 or network.prefixlen != 32:
        raise ValueError("owner ingress must allow exactly one IPv4 address")
    budget = client.request("GET", client.scope + "/providers/Microsoft.Consumption/budgets/" + BUDGET, BUDGET_API)["properties"]
    if budget["amount"] != 10 or budget["timeGrain"] != "Monthly" or budget.get("filter"):
        raise ValueError("unfiltered monthly C$10 resource-group budget required")
    if budget.get("currentSpend", {}).get("unit") != "CAD":
        raise ValueError("CAD budget currency readback required")
    if float(budget["currentSpend"]["amount"]) >= 8:
        raise ValueError("actual spend at or above stop threshold")
    expected = budget_body(client.scope + "/providers/Microsoft.Insights/actionGroups/" + ACTION_GROUP, "placeholder@example.invalid")["properties"]["notifications"]
    for name, notification in expected.items():
        actual = budget["notifications"].get(name, {})
        for field in ("enabled", "operator", "threshold", "thresholdType", "contactGroups"):
            if actual.get(field) != notification[field]:
                raise ValueError("budget notification readback mismatch: " + name + "/" + field)
    scale = app["properties"]["template"]["scale"]
    if scale.get("minReplicas") != 0 or scale.get("maxReplicas") != 1:
        raise ValueError("scale must be min 0, max 1")
    workflow_id = client.scope + "/providers/Microsoft.Logic/workflows/" + WORKFLOW
    workflow = client.request("GET", workflow_id, LOGIC_API)
    if workflow["properties"]["state"] != "Enabled":
        raise ValueError("budget stopper must be enabled")
    expected_definition = workflow_body(client.app, app["location"])["properties"]["definition"]
    if workflow["properties"]["definition"] != expected_definition:
        raise ValueError("budget stopper definition mismatch")
    principal = workflow["identity"]["principalId"]
    role_id = str(uuid.uuid5(uuid.NAMESPACE_URL, client.scope + "/llm-budget-stop-only"))
    role = client.scope + "/providers/Microsoft.Authorization/roleDefinitions/" + role_id
    role_result = client.request("GET", role, "2022-04-01")
    permissions = role_result["properties"]["permissions"]
    if len(permissions) != 1 or set(permissions[0]["actions"]) != {
            "Microsoft.App/containerApps/read", "Microsoft.App/containerApps/stop/action"}:
        raise ValueError("stopper role must permit only app read and stop")
    if any(permissions[0].get(field) for field in ("notActions", "dataActions", "notDataActions")):
        raise ValueError("unexpected stopper role permissions")
    assignment_id = client.app + "/providers/Microsoft.Authorization/roleAssignments/" + str(
        uuid.uuid5(uuid.NAMESPACE_URL, client.app + principal + role_id))
    assignment = client.request("GET", assignment_id, "2022-04-01")["properties"]
    if (assignment["scope"].lower() != client.app.lower() or
            assignment["principalId"] != principal or assignment["roleDefinitionId"].lower() != role_result["id"].lower()):
        raise ValueError("stopper identity assignment must be scoped to this app")
    group = client.request("GET", client.scope + "/providers/Microsoft.Insights/actionGroups/" + ACTION_GROUP,
                           "2023-01-01")["properties"]
    receivers = group.get("logicAppReceivers", [])
    if not group.get("enabled") or len(receivers) != 1 or receivers[0]["resourceId"].lower() != workflow_id.lower():
        raise ValueError("budget stop action group is not wired to the expected workflow")
    callback = client.request("POST", workflow_id + "/triggers/budget/listCallbackUrl", "2016-06-01")["value"]
    if receivers[0]["callbackUrl"] != callback:
        raise ValueError("budget stopper callback mismatch")
    return {"owner_rule_count": 1, "budget_currency": "CAD", "monthly_cap": 10,
            "actual_spend": float(budget["currentSpend"]["amount"]),
            "thresholds": {"actual_alert": 5, "actual_stop": 8, "forecast_stop": 10}}


def test_stopper(client):
    verify_controls(client)
    workflow = client.scope + "/providers/Microsoft.Logic/workflows/" + WORKFLOW
    previous = {run["name"] for run in client.request("GET", workflow + "/runs", LOGIC_API)["value"]}
    callback = client.request("POST", workflow + "/triggers/budget/listCallbackUrl", "2016-06-01")["value"]
    request = urllib.request.Request(callback, data=b'{"phase0_test":true}', headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        if response.status != 202:
            raise ValueError("stopper trigger not accepted")
    runs = []
    for _ in range(36):
        app_state = client.request("GET", client.app, APP_API)["properties"]
        state = app_state.get("runningStatus")
        runs = [run for run in client.request("GET", workflow + "/runs", LOGIC_API)["value"]
                if run["name"] not in previous]
        if runs and runs[0]["properties"]["status"] in {"Failed", "Cancelled", "TimedOut"}:
            raise ValueError("stopper workflow failed; app has not been restarted")
        if (state == "Stopped" and app_state.get("provisioningState") == "Succeeded" and
                runs and runs[0]["properties"]["status"] == "Succeeded"):
            break
        time.sleep(5)
    else:
        raise ValueError("stopper failed to stop app; app has not been restarted")
    client.request("POST", client.app + "/start", APP_API)
    for _ in range(36):
        state = client.request("GET", client.app, APP_API)["properties"].get("runningStatus")
        if state == "Running":
            return {"stopper_run": runs[0]["name"], "stopper_status": "Succeeded",
                    "observed_stopped": True, "observed_restarted": True}
        time.sleep(5)
    raise ValueError("app restart was not observed")


def cooldown(client, image):
    verify_controls(client)
    if not re.fullmatch(r"ghcr\.io/canadasun/llm-serve@sha256:[0-9a-f]{64}", image or ""):
        raise ValueError("the existing preview image must be resolved to an immutable digest")
    app = client.request("GET", client.app, APP_API)
    template = app["properties"]["template"]
    template["scale"].update({"minReplicas": 0, "maxReplicas": 1, "cooldownPeriod": 30})
    template["revisionSuffix"] = "phase0-cooldown"
    template["containers"][0]["image"] = image
    client.request("POST", client.app + "/stop", APP_API)
    for _ in range(36):
        stopped = client.request("GET", client.app, APP_API)["properties"]
        if stopped.get("runningStatus") == "Stopped" and stopped.get("provisioningState") == "Succeeded":
            break
        time.sleep(5)
    else:
        raise ValueError("app stop was not observed; cooldown has not been changed")
    client.request("PATCH", client.app, APP_API, {"location": app["location"], "properties": {
        "configuration": {"activeRevisionsMode": "Single"}, "template": template}})
    for _ in range(60):
        actual = client.request("GET", client.app, APP_API)
        if actual["properties"].get("provisioningState") == "Failed":
            raise ValueError("cooldown revision failed; inspect state before restarting")
        if actual["properties"].get("provisioningState") == "Succeeded":
            break
        time.sleep(5)
    else:
        raise ValueError("cooldown update did not finish; app remains stopped")
    if actual["properties"]["template"]["scale"].get("cooldownPeriod") != 30:
        raise ValueError("cooldown readback failed; app remains stopped")
    client.request("POST", client.app + "/start", APP_API)
    return {"cooldown_seconds": 30, "image": image,
            "revision": actual["properties"].get("latestRevisionName"),
            "research_preview": True, "start_requested": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["allow-ip", "setup-stopper", "verify", "test-stopper", "cooldown"])
    parser.add_argument("--ip")
    parser.add_argument("--image")
    args = parser.parse_args()
    client = Azure()
    if args.action == "allow-ip":
        result = allow_ip(client, args.ip)
    elif args.action == "setup-stopper":
        result = setup_stopper(client)
    elif args.action == "test-stopper":
        result = test_stopper(client)
    elif args.action == "cooldown":
        result = cooldown(client, args.image)
    else:
        result = verify_controls(client)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError) as error:
        raise SystemExit(str(error))
    except Exception:
        raise SystemExit("Lockdown operation failed; inspect scoped Azure state. No secret details printed.")
