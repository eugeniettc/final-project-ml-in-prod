"""Tests for the CT production release orchestration."""

from types import SimpleNamespace

import pytest

from ml import ct_release


RUN_ID = "1" * 32
SHA = "a" * 40
URL = "https://example.invalid"


class FakeClient:
    def __init__(self):
        self.version = "2"
        self.tags = {}

    def get_run(self, run_id):
        assert run_id == RUN_ID
        return SimpleNamespace(
            data=SimpleNamespace(tags={"registry_model_name": "predictmaint-ai"})
        )

    def get_model_version_by_alias(self, name, alias):
        assert name == "predictmaint-ai"
        assert alias == "champion"
        return SimpleNamespace(
            version=self.version,
            status="READY",
            tags=self.tags.copy(),
        )


@pytest.fixture
def environment(monkeypatch):
    client = FakeClient()
    events = []

    monkeypatch.setenv("GITHUB_SHA", SHA)
    monkeypatch.setattr(ct_release, "MlflowClient", lambda: client)
    monkeypatch.setattr(
        ct_release,
        "verify",
        lambda url, version: events.append(("verify", version)),
    )
    monkeypatch.setattr(
        ct_release,
        "validate_production_release_context",
        lambda authorized: events.append(("authorize", authorized)),
    )

    def fake_promote(run_id, execute=False):
        assert run_id == RUN_ID
        events.append(("promote", execute))

        if not execute:
            return {
                "promotion_decision": "ELIGIBLE",
                "promotion_executed": "false",
                "previous_version": "2",
            }

        client.version = "3"
        client.tags = {
            "ct_run_id": RUN_ID,
            "previous_champion_version": "2",
        }

        return {
            "promotion_decision": "ELIGIBLE",
            "promotion_executed": "true",
            "previous_version": "2",
            "new_version": "3",
        }

    monkeypatch.setattr(ct_release, "promote", fake_promote)
    monkeypatch.setattr(
        ct_release,
        "deploy_and_verify",
        lambda url, version, sha: events.append(("deploy", version)),
    )

    return client, events


def test_rejected_candidate_does_not_deploy(monkeypatch, environment):
    client, events = environment

    monkeypatch.setattr(
        ct_release,
        "promote",
        lambda run_id, execute=False: {
            "promotion_decision": "REJECTED",
            "promotion_executed": "false",
            "previous_version": "2",
        },
    )

    ct_release.release(RUN_ID, URL)

    assert client.version == "2"
    assert not any(event[0] == "deploy" for event in events)


def test_eligible_dry_run_does_not_promote(environment):
    client, events = environment

    ct_release.release(RUN_ID, URL, execute=False)

    assert client.version == "2"
    assert ("promote", False) in events
    assert ("promote", True) not in events
    assert not any(event[0] == "deploy" for event in events)


def test_successful_release(environment):
    client, events = environment

    ct_release.release(RUN_ID, URL, execute=True)

    assert client.version == "3"
    assert ("authorize", True) in events
    assert ("promote", True) in events
    assert ("deploy", "3") in events


def test_failed_deployment_triggers_rollback(monkeypatch, environment):
    client, events = environment

    def failed_deploy(url, version, sha):
        raise RuntimeError("Render deployment failed")

    def fake_restore(previous, failed, url, sha, registry):
        events.append(("restore", previous, failed))
        client.version = previous

    monkeypatch.setattr(ct_release, "deploy_and_verify", failed_deploy)
    monkeypatch.setattr(ct_release, "restore", fake_restore)

    with pytest.raises(RuntimeError, match="rollback verified"):
        ct_release.release(RUN_ID, URL, execute=True)

    assert client.version == "2"
    assert ("restore", "2", "3") in events


def test_failed_rollback_is_critical(monkeypatch, environment):
    client, _ = environment

    def failed_deploy(*args):
        raise RuntimeError("Render unavailable")

    def failed_restore(*args):
        raise RuntimeError("Rollback unavailable")

    monkeypatch.setattr(ct_release, "deploy_and_verify", failed_deploy)
    monkeypatch.setattr(ct_release, "restore", failed_restore)

    with pytest.raises(
        RuntimeError,
        match="recovery was not fully verified",
    ):
        ct_release.release(RUN_ID, URL, execute=True)

    assert client.version == "3"


def test_unexpected_champion_change_blocks_rollback(monkeypatch, environment):
    client, events = environment

    def concurrent_change(*args):
        client.version = "4"
        client.tags = {}
        raise RuntimeError("Concurrent registry update")

    monkeypatch.setattr(
        ct_release,
        "deploy_and_verify",
        concurrent_change,
    )

    monkeypatch.setattr(
        ct_release,
        "restore",
        lambda *args: pytest.fail("Must not rollback an unrelated champion"),
    )

    with pytest.raises(RuntimeError, match="unexpected champion change"):
        ct_release.release(RUN_ID, URL, execute=True)

    assert not any(event[0] == "deploy" for event in events)


def test_promotion_failure_preserves_old_champion(monkeypatch, environment):
    client, _ = environment
    original_promote = ct_release.promote

    def failed_promotion(run_id, execute=False):
        if execute:
            raise RuntimeError("Registration failed")
        return original_promote(run_id, execute=False)

    monkeypatch.setattr(ct_release, "promote", failed_promotion)

    with pytest.raises(
        RuntimeError,
        match="champion alias unchanged",
    ):
        ct_release.release(RUN_ID, URL, execute=True)

    assert client.version == "2"


def test_restore_redeploys_previous_champion(
    monkeypatch,
):
    client = FakeClient()
    client.version = "3"
    events = []

    def fake_subprocess(command, check):
        assert "ml.rollback" in command
        assert "--execute" in command
        client.version = "2"

    monkeypatch.setattr(ct_release.subprocess, "run", fake_subprocess)
    monkeypatch.setattr(
        ct_release,
        "deploy_and_verify",
        lambda url, version, sha: events.append((url, version, sha)),
    )

    ct_release.restore("2", "3", URL, SHA, client)

    assert client.version == "2"
    assert events == [(URL, "2", SHA)]
