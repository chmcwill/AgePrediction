# -*- coding: utf-8 -*-
"""
AWS Lambda handler for the Flask app using API Gateway/Lambda Function URL events.
"""
import awsgi

from age_prediction.app import create_app

# WSGI app cached at import time to reduce cold-start overhead.
app = create_app()


def handler(event, context):
    """Lambda entrypoint (inputs: event/context; output: API response dict)."""
    # awsgi adapts API Gateway/Lambda Function URL events to WSGI.
    return awsgi.response(app, event, context)
