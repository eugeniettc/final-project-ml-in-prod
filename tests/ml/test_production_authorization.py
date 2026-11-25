
import pytest

from ml.promote_candidate import (
    validate_production_release_context,
)


def test_production_promotion_denied_by_default():
    with pytest.raises(
        RuntimeError,
        match="explicit authorization",
    ):
        validate_production_release_context(False)


def test_production_promotion_denied_outside_github(
    monkeypatch,
):
    for key in [
        "GITHUB_REPOSITORY",
        "GITHUB_REF",
        "GITHUB_EVENT_NAME",
        "CT_ALLOW_PRODUCTION_PROMOTION",
    ]:
        monkeypatch.delenv(key, raising=False)

    with pytest.raises(RuntimeError):
        validate_production_release_context(True)


def test_production_promotion_denied_from_dev(
    monkeypatch,
):
    monkeypatch.setenv(
        "GITHUB_REPOSITORY",
        "eugeniettc/final-project-ml-in-prod",
    )
    monkeypatch.setenv("GITHUB_REF", "refs/heads/dev")
    monkeypatch.setenv(
        "GITHUB_EVENT_NAME", "workflow_dispatch"
    )
    monkeypatch.setenv(
        "CT_ALLOW_PRODUCTION_PROMOTION", "true"
    )

    with pytest.raises(
        RuntimeError, match="GITHUB_REF"
    ):
        validate_production_release_context(True)


def test_production_promotion_authorized_on_main(
    monkeypatch,
):
    monkeypatch.setenv(
        "GITHUB_REPOSITORY",
        "eugeniettc/final-project-ml-in-prod",
    )
    monkeypatch.setenv(
        "GITHUB_REF", "refs/heads/main"
    )
    monkeypatch.setenv(
        "GITHUB_EVENT_NAME", "workflow_dispatch"
    )
    monkeypatch.setenv(
        "CT_ALLOW_PRODUCTION_PROMOTION", "true"
    )

    validate_production_release_context(True)
