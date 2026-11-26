# SkillGap

Scores how well a resume matches a job posting (0–100) and lists the skills the posting asks for that the resume is missing.

> Work in progress. Done: matching core, tests, CI. Next: FastAPI endpoint, AWS deploy, CD.

## How it works
- **Skill coverage (70%)**: Finds known skills in both texts using a list of skills and their synonyms (`src/skillgap/skills.py`), so "JS" counts as "JavaScript" and "k8s" counts as "Kubernetes". The score is the share of the posting's skills that appear in the resume.
- **TF-IDF similarity (30%)**: Cosine similarity of the two texts after synonyms are normalized, using scikit-learn.

```python
from skillgap import match

result = match(resume_text, posting_text)
result.score           # 81
result.missing_skills  # ["aws", "docker"]
```

## Run locally
```bash
pip install -e ".[dev]"
ruff check .
pytest --cov=skillgap
```

CI (GitHub Actions) runs lint and tests on Python 3.11 and 3.12 on every push.
