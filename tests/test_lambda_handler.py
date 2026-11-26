"""Checks the Lambda entry point with fake API Gateway (HTTP API v2) events,
so the AWS wiring is tested in CI without deploying anything."""

import json

from skillgap.lambda_handler import handler


def make_event(method: str, path: str, body: dict | None = None) -> dict:
    return {
        "version": "2.0",
        "routeKey": "$default",
        "rawPath": path,
        "rawQueryString": "",
        "headers": {"content-type": "application/json", "host": "example.com"},
        "requestContext": {
            "http": {
                "method": method,
                "path": path,
                "protocol": "HTTP/1.1",
                "sourceIp": "127.0.0.1",
                "userAgent": "pytest",
            },
            "stage": "$default",
        },
        "body": json.dumps(body) if body is not None else None,
        "isBase64Encoded": False,
    }


class FakeContext:
    function_name = "skillgap"
    aws_request_id = "test"


def test_health_through_lambda():
    response = handler(make_event("GET", "/health"), FakeContext())
    assert response["statusCode"] == 200
    assert json.loads(response["body"]) == {"status": "ok"}


def test_match_through_lambda():
    body = {"resume_text": "Python and SQL", "posting_text": "Need Python, SQL, AWS"}
    response = handler(make_event("POST", "/match", body), FakeContext())
    assert response["statusCode"] == 200
    assert json.loads(response["body"])["missing_skills"] == ["aws"]


def test_bad_input_through_lambda():
    response = handler(make_event("POST", "/match", {"resume_text": ""}), FakeContext())
    assert response["statusCode"] == 422
