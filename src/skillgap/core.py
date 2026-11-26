"""Resume / job-posting matching.

The score combines two signals:
  * skill coverage: the share of skills named in the posting that also appear
    in the resume (synonyms count, e.g. "JS" == "JavaScript")
  * TF-IDF cosine similarity of the two texts, after synonyms are normalized

Coverage carries most of the weight because it is what recruiters and ATS
filters look at; TF-IDF adds a smaller signal for overall wording overlap.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from skillgap.skills import SKILLS

COVERAGE_WEIGHT = 0.7
TFIDF_WEIGHT = 0.3


@dataclass(frozen=True)
class MatchResult:
    score: int                      # 0-100
    skill_coverage: float | None    # 0-1, None if the posting names no known skills
    tfidf_similarity: float         # 0-1
    matched_skills: list[str]
    missing_skills: list[str]


def _alias_pattern(alias: str) -> str:
    # Let spaces in an alias match any whitespace or a hyphen ("machine-learning").
    return re.escape(alias).replace(r"\ ", r"[\s\-]+")


def _build_index() -> tuple[re.Pattern[str], dict[str, list[str]]]:
    alias_to_skills: dict[str, list[str]] = {}
    for skill, aliases in SKILLS.items():
        for alias in aliases:
            alias_to_skills.setdefault(alias.lower(), []).append(skill)
    # Longest aliases first so "react.js" wins over "react", "asp.net" over ".net".
    ordered = sorted(alias_to_skills, key=len, reverse=True)
    body = "|".join(_alias_pattern(a) for a in ordered)
    # Not preceded or followed by a word character, '+' or '#', so "java" never
    # matches inside "javascript" and "c++" stays distinct from "c".
    pattern = re.compile(rf"(?<![\w+#.])(?:{body})(?![\w+#])", re.IGNORECASE)
    return pattern, alias_to_skills


_PATTERN, _ALIAS_TO_SKILLS = _build_index()


def _skills_for(match: re.Match[str]) -> list[str]:
    key = re.sub(r"[\s\-]+", " ", match.group(0).lower())
    return _ALIAS_TO_SKILLS.get(key, [])


def extract_skills(text: str) -> set[str]:
    """Return the canonical skills mentioned in ``text``."""
    found: set[str] = set()
    for m in _PATTERN.finditer(text):
        found.update(_skills_for(m))
    return found


def _skill_token(skill: str) -> str:
    safe = skill.replace("+", "plus").replace("#", "sharp").replace(".", "dot")
    return "skill_" + re.sub(r"\W+", "_", safe)


def _normalize(text: str) -> str:
    """Replace every skill alias with one token per canonical skill."""
    return _PATTERN.sub(
        lambda m: " " + " ".join(_skill_token(s) for s in _skills_for(m)) + " ",
        text,
    )


def _tfidf_similarity(resume: str, posting: str) -> float:
    vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
    try:
        matrix = vectorizer.fit_transform([_normalize(resume), _normalize(posting)])
    except ValueError:  # nothing left after stop-word removal
        return 0.0
    return float(cosine_similarity(matrix[0], matrix[1])[0, 0])


def match(resume_text: str, posting_text: str) -> MatchResult:
    """Score how well a resume matches a job posting.

    Raises ValueError if either input is empty or only whitespace.
    """
    if not resume_text or not resume_text.strip():
        raise ValueError("resume_text is empty")
    if not posting_text or not posting_text.strip():
        raise ValueError("posting_text is empty")

    resume_skills = extract_skills(resume_text)
    posting_skills = extract_skills(posting_text)
    matched = sorted(posting_skills & resume_skills)
    missing = sorted(posting_skills - resume_skills)

    similarity = _tfidf_similarity(resume_text, posting_text)

    if posting_skills:
        coverage: float | None = len(matched) / len(posting_skills)
        raw = COVERAGE_WEIGHT * coverage + TFIDF_WEIGHT * similarity
    else:
        coverage = None
        raw = similarity

    return MatchResult(
        score=round(100 * min(max(raw, 0.0), 1.0)),
        skill_coverage=coverage,
        tfidf_similarity=similarity,
        matched_skills=matched,
        missing_skills=missing,
    )
