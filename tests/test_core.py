import pytest

from skillgap import extract_skills, match

RESUME = """
Data science intern. Trained and evaluated ML models in Python with PyTorch and
scikit-learn; built embedding-based evaluation and fairness metrics. Wrote SQL
queries and built Power BI dashboards. Built a Python automation pipeline.
Coursework: data structures, algorithms, Java, C/C++.
"""

GOOD_POSTING = """
Data Science Intern. Requirements: Python, SQL, machine learning, PyTorch or
scikit-learn. Experience with Power BI dashboards and data structures a plus.
"""

POOR_POSTING = """
Frontend Engineer. Build UIs with React, TypeScript, HTML and CSS. Experience
with GraphQL and Figma-based design handoff. Deploy with Docker and Kubernetes.
"""


# ---- skill extraction -------------------------------------------------------

@pytest.mark.parametrize(
    "text, skill",
    [
        ("Proficient in JS", "javascript"),
        ("Experience with ML", "machine learning"),
        ("Deployed on k8s", "kubernetes"),
        ("Built with sklearn", "scikit-learn"),
        ("Hosted on Amazon Web Services", "aws"),
        ("machine-learning pipelines", "machine learning"),
        ("PYTHON", "python"),
    ],
)
def test_synonyms_map_to_canonical_skill(text, skill):
    assert skill in extract_skills(text)


@pytest.mark.parametrize(
    "text, skill",
    [
        ("Skills: Python.", "python"),
        ("databases (SQL)", "sql"),
        ("C++, Java", "c++"),
        ("C#/.NET developer", "c#"),
        ("C#/.NET developer", ".net"),
        ("Node.js backend", "node.js"),
        ("CI/CD pipelines", "ci/cd"),
    ],
)
def test_skills_found_next_to_punctuation(text, skill):
    assert skill in extract_skills(text)


# Skills found in a real resume and a real posting during manual testing.
@pytest.mark.parametrize(
    "text, skill",
    [
        ("Applied statistical methods to validate outputs", "statistics"),
        ("statistical arbitrage research", "statistics"),
        ("facial recognition pipeline", "computer vision"),
        ("image-processing optimizations", "computer vision"),
        ("embedding-based evaluation", "embeddings"),
        ("demand-forecasting model", "forecasting"),
        ("Introduction to Data Science", "data science"),
        ("Hugging Face, OpenAI API", "hugging face"),
        ("Hugging Face, OpenAI API", "openai api"),
        ("AWS (Lambda, API Gateway, ECR)", "api gateway"),
        ("AWS (Lambda, API Gateway, ECR)", "ecr"),
        ("Java/JDBC backend over an Oracle database", "jdbc"),
        ("Java/JDBC backend over an Oracle database", "oracle"),
        ("Designed a relational database schema", "relational databases"),
        ("Jira, Confluence, GitHub, Sonar, Artifactory, and Datadog", "jira"),
        ("Jira, Confluence, GitHub, Sonar, Artifactory, and Datadog", "confluence"),
        ("Jira, Confluence, GitHub, Sonar, Artifactory, and Datadog", "sonarqube"),
        ("Jira, Confluence, GitHub, Sonar, Artifactory, and Datadog", "artifactory"),
        ("Jira, Confluence, GitHub, Sonar, Artifactory, and Datadog", "datadog"),
        ("process improvement, automation, and configuration as code", "infrastructure as code"),
    ],
)
def test_skills_from_real_applications(text, skill):
    assert skill in extract_skills(text)


def test_research_experimentation_is_not_ab_testing():
    # Bug found on a real resume: "experimentation" used to count as A/B testing.
    assert "a/b testing" not in extract_skills("reducing bias through systematic experimentation")


def test_c_slash_cpp_counts_as_both():
    assert {"c", "c++"} <= extract_skills("C/C++")


@pytest.mark.parametrize(
    "text, not_skill",
    [
        ("JavaScript developer", "java"),
        ("reactive programming", "react"),
        ("MySQL database", "sql"),
        ("NoSQL stores", "sql"),
        ("GitHub profile", "git"),
        ("ready to go above and beyond", "go"),
    ],
)
def test_no_false_positives_inside_other_words(text, not_skill):
    assert not_skill not in extract_skills(text)


# ---- matching ---------------------------------------------------------------

def test_known_good_match_scores_high():
    result = match(RESUME, GOOD_POSTING)
    assert result.score >= 70
    assert result.missing_skills == []


def test_known_poor_match_scores_low():
    result = match(RESUME, POOR_POSTING)
    assert result.score <= 30


def test_good_match_beats_poor_match():
    assert match(RESUME, GOOD_POSTING).score > match(RESUME, POOR_POSTING).score


def test_missing_skill_detection():
    posting = "We need Python, SQL, AWS and Docker experience."
    result = match(RESUME, posting)
    assert result.matched_skills == ["python", "sql"]
    assert result.missing_skills == ["aws", "docker"]
    assert result.skill_coverage == pytest.approx(0.5)


def test_synonym_in_resume_satisfies_posting():
    result = match("Built web apps in JS and TS.", "Requires JavaScript and TypeScript.")
    assert result.missing_skills == []
    assert result.skill_coverage == 1.0


def test_identical_text_scores_100():
    assert match(GOOD_POSTING, GOOD_POSTING).score == 100


def test_posting_with_no_known_skills():
    result = match(RESUME, "Friendly, motivated team player wanted.")
    assert result.skill_coverage is None
    assert result.missing_skills == []
    assert 0 <= result.score <= 100


@pytest.mark.parametrize("bad", ["", "   ", "\n\t"])
def test_empty_resume_raises(bad):
    with pytest.raises(ValueError):
        match(bad, GOOD_POSTING)


@pytest.mark.parametrize("bad", ["", "   ", "\n\t"])
def test_empty_posting_raises(bad):
    with pytest.raises(ValueError):
        match(RESUME, bad)


def test_only_stop_words_does_not_crash():
    result = match("the and of", "a an the")
    assert result.score == 0


def test_very_long_inputs():
    result = match(RESUME * 500, GOOD_POSTING * 500)
    assert result.score >= 70


def test_score_is_deterministic():
    assert match(RESUME, GOOD_POSTING) == match(RESUME, GOOD_POSTING)


def test_score_always_in_range():
    for posting in (GOOD_POSTING, POOR_POSTING, "Python"):
        assert 0 <= match(RESUME, posting).score <= 100
