def test_app_configured(app):
    assert app is not None
    assert app.config["UPLOAD_FOLDER"] == "static/images"
    import age_prediction.app as app_module

    assert app.config["MAX_CONTENT_LENGTH"] == app_module.max_mb * 1024 * 1024


def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.is_json
    assert response.get_json().get("ok") is True


def test_404_handler(client):
    response = client.get("/does-not-exist")
    assert response.status_code == 404
