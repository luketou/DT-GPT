import json
import urllib.error

import pytest

from pipeline.vllm_client import (
    VLLMEndpointError,
    build_vllm_url,
    check_vllm_endpoint,
)


def test_build_vllm_url_appends_path_once():
    assert build_vllm_url("http://127.0.0.1:18101/v1/", "models") == "http://127.0.0.1:18101/v1/models"
    assert build_vllm_url("http://127.0.0.1:18101/v1", "/completions") == "http://127.0.0.1:18101/v1/completions"


def test_check_vllm_endpoint_wraps_connection_failure(monkeypatch):
    def fail_urlopen(request, timeout):
        raise urllib.error.URLError(ConnectionRefusedError(111, "Connection refused"))

    monkeypatch.setattr("urllib.request.urlopen", fail_urlopen)

    with pytest.raises(VLLMEndpointError) as excinfo:
        check_vllm_endpoint("http://127.0.0.1:18101/v1/", timeout=1)

    message = str(excinfo.value)
    assert "vLLM endpoint is not reachable" in message
    assert "http://127.0.0.1:18101/v1/models" in message
    assert "Connection refused" in message


def test_check_vllm_endpoint_accepts_models_response(monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps({"data": [{"id": "local-model"}]}).encode("utf-8")

    def ok_urlopen(request, timeout):
        assert request.full_url == "http://127.0.0.1:18101/v1/models"
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", ok_urlopen)

    assert check_vllm_endpoint("http://127.0.0.1:18101/v1/", timeout=1) == ["local-model"]
