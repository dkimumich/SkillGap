# SkillGap

![CI/CD](https://github.com/dkimumich/SkillGap/actions/workflows/ci.yml/badge.svg)

Scores how well a resume matches a job posting (0–100) and lists the skills the posting asks for that the resume is missing.

**Try it live:** https://fsx5xni654.execute-api.us-east-2.amazonaws.com
Open **POST /match** → **Try it out** → paste a resume and a job posting → **Execute**. (The first request after a quiet period takes a few seconds while AWS Lambda starts up.)

## Why

During my internship application season, keyword-matching tools often gave me inaccurate scores. I built SkillGap to understand how resume–job matching works, measure how accurate it is.

## Architecture

- **Serving a request:** API Gateway (public URL, rate-limited) → AWS Lambda running a Docker image → Mangum → FastAPI → `match()`.
- **Shipping code:** every push runs lint and tests in GitHub Actions. Pushes to `main` that pass are built, pushed to Amazon ECR and deployed to Lambda automatically. GitHub signs in to AWS with OIDC, so no AWS keys are stored in the repo or in CI.

## How matching works

- **Skill coverage (70%):** finds known skills in both texts using a vocabulary of about 90 skills and their synonyms (`src/skillgap/skills.py`), so "JS" counts as JavaScript and "k8s" as Kubernetes. Word-boundary rules stop false matches like "Java" inside "JavaScript". Coverage = the share of the posting's skills found in the resume.
- **TF-IDF similarity (30%):** cosine similarity of the two texts after synonyms are normalized, using scikit-learn.

## API

| Route | Method | Description |
| --- | --- | --- |
| `/match` | POST | Score a resume against a posting |
| `/health` | GET | Health check |
| `/docs` | GET | Interactive documentation |

```bash
curl -X POST https://fsx5xni654.execute-api.us-east-2.amazonaws.com/match \
  -H "Content-Type: application/json" \
  -d '{"resume_text": "Python and SQL developer", "posting_text": "Need Python, SQL, AWS"}'
```

```json
{
  "score": 59,
  "skill_coverage": 0.667,
  "tfidf_similarity": 0.411,
  "matched_skills": ["python", "sql"],
  "missing_skills": ["aws"]
}
```

Invalid input (empty, missing or over 50,000 characters) returns `422`.

## Tech stack

Python · scikit-learn · FastAPI · Pydantic · pytest · ruff · Docker · AWS Lambda · API Gateway · ECR · IAM · GitHub Actions (CI/CD with OIDC)

## Testing

52 pytest tests with 100% coverage: skill extraction and synonyms, scoring, edge cases (empty and very long input), API validation, and the Lambda handler with simulated API Gateway events.

## Run locally

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows (macOS/Linux: source .venv/bin/activate)
pip install -e ".[dev]"
ruff check .
pytest --cov=skillgap
uvicorn skillgap.api:app --reload  # then open http://127.0.0.1:8000/docs
```

Manual AWS deployment steps are in [docs/DEPLOY.md](docs/DEPLOY.md).

## Project structure

```
src/skillgap/
  skills.py           skill vocabulary and synonyms
  core.py             matching logic
  api.py              FastAPI app
  lambda_handler.py   AWS Lambda entry point (Mangum)
tests/                pytest suite
infra/                IAM trust and permission policies
Dockerfile            Lambda container image
.github/workflows/    CI/CD pipeline
```

## Known limitations

- A posting that names only one known skill can score high on coverage alone.
- Only finds skills that are named; it does not infer skills from descriptions of work.
- Accepts plain text only (no PDF upload yet).

## Roadmap

- [x] Synonym-aware skill extraction and match scoring
- [x] pytest suite and GitHub Actions CI
- [x] FastAPI endpoint (`POST /match`)
- [x] Deploy to AWS (Docker → ECR → Lambda + API Gateway)
- [x] Continuous deployment on merge to `main` (GitHub OIDC)
- [ ] Evaluate on real applications: TF-IDF vs. embedding-based matching
- [ ] Reduce the score for postings that name very few skills
- [ ] Simple web front end for non-technical users
- [ ] Chrome extension to log applications to Google Sheets
- [ ] Suggested resume bullet rewrites
