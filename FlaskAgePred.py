# -*- coding: utf-8 -*-
"""
Compatibility shim that exposes the Flask app via the new factory.
"""

from age_prediction.app import create_app  # noqa: F401

app = create_app()


if __name__ == "__main__":
    app.run(debug=False)
