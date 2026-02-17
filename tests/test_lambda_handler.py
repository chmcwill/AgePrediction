import pytest

import lambda_handler


def test_handler_returns_awsgi_response(monkeypatch):
    monkeypatch.setattr(lambda_handler, "app", object())
    monkeypatch.setattr(lambda_handler.awsgi, "response", lambda *_args, **_kwargs: {"ok": True})
    result = lambda_handler.handler({"event": "value"}, {"ctx": "value"})
    assert result == {"ok": True}


def test_handler_reraises_exceptions(monkeypatch):
    def _raise(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(lambda_handler.awsgi, "response", _raise)
    with pytest.raises(RuntimeError):
        lambda_handler.handler({"event": "value"}, {"ctx": "value"})
