import json
import os
from unittest.mock import Mock

import pytest

from deploy import azure_guard, lockdown


def test_policy_fails_closed_for_other_resources(monkeypatch):
    for key, value in (("RG", "another-rg"), ("APP", "another-app")):
        with monkeypatch.context() as context:
            context.setenv(key, value)
            with pytest.raises(ValueError, match="restricted"):
                azure_guard.policy()


def test_unapproved_rollout_makes_no_azure_calls(monkeypatch):
    command = Mock(side_effect=AssertionError("must remain offline"))
    monkeypatch.setattr(azure_guard, "az", command)
    with pytest.raises(ValueError, match="go-live approval"):
        azure_guard.check_rollout()
    command.assert_not_called()


def test_management_requests_cannot_escape_resource_group(monkeypatch):
    monkeypatch.setenv("SUBSCRIPTION", "00000000-0000-0000-0000-000000000000")
    client = azure_guard.Azure()
    for resource in (client.scope + "-other/providers/Microsoft.App/containerApps/llm-serve",
                     "/subscriptions/other/resourceGroups/other/providers/Microsoft.App/containerApps/llm-serve"):
        with pytest.raises(ValueError, match="outside approved"):
            client.request("POST", resource, "2025-07-01")


def test_stop_target_is_fixed_and_uses_managed_identity():
    app = "/subscriptions/example/resourceGroups/llm-from-scratch-rg/providers/Microsoft.App/containerApps/llm-serve"
    workflow = lockdown.workflow_body(app, "canadacentral")
    actions = workflow["properties"]["definition"]["actions"]
    assert list(actions) == ["stop_llm"]
    assert actions["stop_llm"]["inputs"]["uri"] == "https://management.azure.com" + app + "/stop?api-version=2025-07-01"
    assert actions["stop_llm"]["inputs"]["authentication"]["type"] == "ManagedServiceIdentity"
    assert "triggerBody" not in json.dumps(actions)


def test_budget_alert_does_not_stop_and_both_stop_thresholds_do():
    notifications = lockdown.budget_body("scoped-group", "owner@example.invalid")["properties"]["notifications"]
    assert notifications["actual_alert"]["contactGroups"] == []
    for name, threshold, kind in (("actual_alert", 50, "Actual"),
                                   ("actual_stop", 80, "Actual"),
                                   ("forecast_stop", 100, "Forecasted")):
        assert notifications[name]["threshold"] == threshold
        assert notifications[name]["thresholdType"] == kind
        assert notifications[name]["operator"] == "GreaterThanOrEqualTo"
    assert notifications["actual_stop"]["contactGroups"] == ["scoped-group"]
    assert notifications["forecast_stop"]["contactGroups"] == ["scoped-group"]


def test_owner_ip_update_rejects_networks_and_replaces_all_rules():
    client = Mock()
    rules = [{"name": "owner", "action": "Allow", "ipAddressRange": "192.0.2.1/32",
              "description": "Owner access only"}]
    client.request.return_value = {"properties": {"configuration": {"ingress": {"ipSecurityRestrictions": rules}}}}
    assert lockdown.allow_ip(client, "192.0.2.1")["owner_prefix_length"] == 32
    assert client.request.call_args_list[0].args[3]["properties"]["configuration"]["ingress"]["ipSecurityRestrictions"] == rules
    with pytest.raises(ValueError):
        lockdown.allow_ip(client, "192.0.2.1/24")
