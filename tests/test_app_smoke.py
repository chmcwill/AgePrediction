def test_app_configured(app):
    assert app is not None
    assert app.config["UPLOAD_FOLDER"] == "static/images"
    import age_prediction.app as app_module

    assert app.config["MAX_CONTENT_LENGTH"] == app_module.max_mb * 1024 * 1024


def test_home_page_renders(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"<" in response.data  # basic HTML returned


def test_resultspage_redirects_without_upload(client):
    response = client.get("/resultspage", follow_redirects=False)
    assert response.status_code == 302
    assert response.headers["Location"].endswith("/")


def test_404_handler(client):
    response = client.get("/does-not-exist")
    assert response.status_code == 404
