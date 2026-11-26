import pytest

from skillgap import extract_skills, match, required_skills

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


def test_identical_text_scores_very_high():
    # Not exactly 100: coverage is shrunk slightly toward 50% (see few-skills tests).
    result = match(GOOD_POSTING, GOOD_POSTING)
    assert result.skill_coverage == 1.0
    assert result.score >= 90


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


# ---- required vs. mentioned skills -----------------------------------------

SECTIONED_POSTING = """
About the team
We build internal tools with Jira, Confluence and Datadog.

Responsibilities
- Automate workflows and maintain GitHub integrations

Minimum Qualifications
- Experience with Python
- Knowledge of agile practices

Preferred Qualifications
- Docker experience
"""


def test_required_skills_come_from_qualification_sections():
    assert required_skills(SECTIONED_POSTING) == {"python", "agile"}


def test_preferred_section_is_not_required():
    assert "docker" not in required_skills(SECTIONED_POSTING)


def test_posting_without_headings_has_no_required_skills():
    assert required_skills("We need Python, SQL, AWS and Docker experience.") == set()


@pytest.mark.parametrize(
    "line",
    ["• Strong communication skills", "- Required: Python", "Excellent communication skills."],
)
def test_bullets_and_sentences_are_not_headings(line):
    posting = "Responsibilities\n" + line + " SQL\n"
    assert required_skills(posting) == set()


@pytest.mark.parametrize("heading", ["## Requirements", "**Requirements**", "Requirements:"])
def test_markdown_and_colon_headings(heading):
    assert required_skills(heading + "\nPython and SQL\n") == {"python", "sql"}


def test_missing_required_lists_only_required_gaps():
    result = match("Agile team member using Jira.", SECTIONED_POSTING)
    assert result.missing_required == ["python"]
    assert "datadog" in result.missing_skills


def test_required_skills_outweigh_mentioned_tools():
    has_required = match("Python developer on an agile team.", SECTIONED_POSTING)
    has_tools_only = match("Used Jira, Confluence, Datadog, GitHub and Docker.", SECTIONED_POSTING)
    assert has_required.score > has_tools_only.score


def test_coverage_weighting():
    # Required: python, agile (both matched). Mentioned: jira, confluence, datadog, github,
    # docker (github matched = 1 of 5). Coverage = 0.75 * 1.0 + 0.25 * 0.2 = 0.8
    result = match("Python, agile, GitHub.", SECTIONED_POSTING)
    assert result.skill_coverage == pytest.approx(0.8)


# ---- few-skills postings -----------------------------------------------------

FULL_RESUME = "Python developer with SQL, AWS, Docker, React, Java, Git, Linux, pandas, Kubernetes"
TEN_SKILL_POSTING = "We need Python, SQL, AWS, Docker, React, Java, Git, Linux, pandas, Kubernetes."


def test_one_of_one_scores_lower_than_ten_of_ten():
    one = match(FULL_RESUME, "We need Python.")
    ten = match(FULL_RESUME, TEN_SKILL_POSTING)
    assert one.skill_coverage == ten.skill_coverage == 1.0  # raw coverage is unchanged
    assert one.score < ten.score
    assert one.score <= 60


def test_more_evidence_moves_coverage_further_from_neutral():
    from skillgap.core import _adjusted_coverage

    assert _adjusted_coverage(1.0, 1) == pytest.approx(2 / 3)
    assert _adjusted_coverage(1.0, 10) == pytest.approx(11 / 12)
    assert _adjusted_coverage(0.0, 1) == pytest.approx(1 / 3)
    assert _adjusted_coverage(0.5, 3) == pytest.approx(0.5)


@pytest.mark.parametrize(
    "posting, level",
    [
        ("Friendly team player wanted.", "low"),
        ("We need Python.", "low"),
        ("Python, SQL and AWS.", "medium"),
        (TEN_SKILL_POSTING, "high"),
    ],
)
def test_confidence_levels(posting, level):
    assert match(FULL_RESUME, posting).confidence == level
