import pytest
from fastapi.testclient import TestClient

from skillgap.api import MAX_CHARS, app

client = TestClient(app)

RESUME = "Python, SQL and machine learning with PyTorch. Built Power BI dashboards."
POSTING = "Data intern: Python, SQL, machine learning, AWS."


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_match_returns_score_and_skills():
    response = client.post("/match", json={"resume_text": RESUME, "posting_text": POSTING})
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["score"] <= 100
    assert body["matched_skills"] == ["machine learning", "python", "sql"]
    assert body["missing_skills"] == ["aws"]
    assert body["skill_coverage"] == pytest.approx(0.75)


def test_response_has_expected_fields():
    response = client.post("/match", json={"resume_text": RESUME, "posting_text": POSTING})
    assert set(response.json()) == {
        "score", "skill_coverage", "tfidf_similarity", "matched_skills", "missing_skills",
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"resume_text": "", "posting_text": POSTING},          # empty
        {"resume_text": RESUME, "posting_text": ""},           # empty
        {"resume_text": "   ", "posting_text": POSTING},       # whitespace only
        {"resume_text": RESUME},                               # missing field
        {"resume_text": 123, "posting_text": POSTING},         # wrong type
        {"resume_text": "a" * (MAX_CHARS + 1), "posting_text": POSTING},  # too long
    ],
)
def test_bad_input_returns_422(payload):
    response = client.post("/match", json=payload)
    assert response.status_code == 422


def test_get_on_match_not_allowed():
    assert client.get("/match").status_code == 405
