
import json
import sys
from urllib.parse import parse_qs, urlsplit

import pytest

from ml import render_redeploy


SHA = "a" * 40


class FakeResponse:
    def __init__(self, payload):
        self.status = 200
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self, *args):
        return json.dumps(self.payload).encode()


def configure_render(monkeypatch):
    monkeypatch.setenv(
        "RENDER_PRODUCTION_DEPLOY_HOOK_URL",
        "https://api.render.com/deploy/test?key=fake",
    )
    monkeypatch.setenv("RENDER_API_KEY", "fake-api-key")
    monkeypatch.setenv(
        "RENDER_PRODUCTION_SERVICE_ID", "srv-test"
    )


def test_dry_run_never_deploys(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "render_redeploy",
            "--url",
            "https://example.com",
            "--expected-version",
            "2",
        ],
    )

    calls = []

    monkeypatch.setattr(
        render_redeploy,
        "verify",
        lambda url, version: calls.append(
            (url, version)
        ),
    )

    monkeypatch.setattr(
        render_redeploy,
        "deploy_and_verify",
        lambda *args: pytest.fail(
            "Deployment must not be triggered"
        ),
    )

    render_redeploy.main()

    assert calls == [("https://example.com", "2")]


def test_successful_deployment(monkeypatch):
    configure_render(monkeypatch)

    requests = []
    verified = []

    def fake_urlopen(request, timeout=30):
        requests.append(request)

        if "/deploy/test" in request.full_url:
            return FakeResponse({
                "deploy": {"id": "dep-test"}
            })

        return FakeResponse({
            "status": "live",
            "commit": {"id": SHA},
        })

    monkeypatch.setattr(
        render_redeploy, "urlopen", fake_urlopen
    )

    monkeypatch.setattr(
        render_redeploy,
        "verify",
        lambda url, version: verified.append(
            (url, version)
        ) or {"failure_prediction": 0},
    )

    render_redeploy.deploy_and_verify(
        "https://example.com", "3", SHA
    )

    query = parse_qs(
        urlsplit(requests[0].full_url).query
    )

    assert query["ref"] == [SHA]
    assert verified == [("https://example.com", "3")]


def test_wrong_commit_is_rejected(monkeypatch):
    configure_render(monkeypatch)

    def fake_urlopen(request, timeout=30):
        if "/deploy/test" in request.full_url:
            return FakeResponse({
                "deploy": {"id": "dep-test"}
            })

        return FakeResponse({
            "status": "live",
            "commit": {"id": "b" * 40},
        })

    monkeypatch.setattr(
        render_redeploy, "urlopen", fake_urlopen
    )

    monkeypatch.setattr(
        render_redeploy,
        "verify",
        lambda *args: pytest.fail(
            "Wrong commit must never be verified"
        ),
    )

    with pytest.raises(
        RuntimeError, match="Wrong Git commit"
    ):
        render_redeploy.deploy_and_verify(
            "https://example.com", "3", SHA
        )


def test_failed_deployment_is_rejected(monkeypatch):
    configure_render(monkeypatch)

    def fake_urlopen(request, timeout=30):
        if "/deploy/test" in request.full_url:
            return FakeResponse({
                "deploy": {"id": "dep-test"}
            })

        return FakeResponse({
            "status": "build_failed",
        })

    monkeypatch.setattr(
        render_redeploy, "urlopen", fake_urlopen
    )

    with pytest.raises(
        RuntimeError, match="Render deployment failed"
    ):
        render_redeploy.deploy_and_verify(
            "https://example.com", "3", SHA
        )
