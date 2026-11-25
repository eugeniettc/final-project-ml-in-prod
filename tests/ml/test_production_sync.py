from types import SimpleNamespace
import pytest
from ml import check_production_sync


URL = "https://example.com"


class FakeRegistry:
    def __init__(self, versions=("2", "2")):
        self.versions = iter(versions)

    def get_model_version_by_alias(self, name, alias):
        assert name == "predictmaint-ai"
        assert alias == "champion"
        return SimpleNamespace(
            version=next(self.versions),
            status="READY",
        )


def test_synchronized_production(monkeypatch):
    registry = FakeRegistry()

    monkeypatch.setattr(
        check_production_sync,
        "verify",
        lambda url, version: {"failure_prediction": 0},
    )

    result = check_production_sync.check_production_sync(URL, registry)

    assert result == "2"


def test_wrong_render_version_is_detected(monkeypatch):
    registry = FakeRegistry()

    def fail_verification(url, version):
        raise AssertionError("Render serves v1, expected v2")

    monkeypatch.setattr(
        check_production_sync,
        "verify",
        fail_verification,
    )

    with pytest.raises(RuntimeError, match="PRODUCTION SYNC FAILED"):
        check_production_sync.check_production_sync(URL, registry)


def test_champion_change_is_detected(monkeypatch):
    registry = FakeRegistry(("2", "3"))

    monkeypatch.setattr(
        check_production_sync,
        "verify",
        lambda url, version: {"failure_prediction": 0},
    )

    with pytest.raises(RuntimeError, match="changed during verification"):
        check_production_sync.check_production_sync(URL, registry)
