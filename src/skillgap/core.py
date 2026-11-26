"""Resume / job-posting matching.

The score combines two signals:
  * skill coverage: the share of skills named in the posting that also appear
    in the resume (synonyms count, e.g. "JS" == "JavaScript")
  * TF-IDF cosine similarity of the two texts, after synonyms are normalized

Coverage carries most of the weight because it is what recruiters and ATS
filters look at; TF-IDF adds a smaller signal for overall wording overlap.

Postings are split into sections by their headings. Skills under a
"Qualifications" / "Requirements" heading are *required*; skills found only
elsewhere (responsibilities, tools the team uses, "nice to have") are
*mentioned*. Required skills carry most of the coverage weight.

When a posting names only a few known skills, coverage is weak evidence
(1 of 1 is not as convincing as 10 of 10), so the score pulls coverage toward
a neutral 50% as if the posting also named two "unknown" skills. The reported
skill_coverage stays the raw value; only the score is adjusted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from skillgap.skills import SKILLS

COVERAGE_WEIGHT = 0.7
TFIDF_WEIGHT = 0.3
REQUIRED_WEIGHT = 0.75   # share of coverage that comes from required skills
MENTIONED_WEIGHT = 0.25  # share from skills only mentioned elsewhere
PRIOR_COVERAGE = 0.5     # neutral coverage assumed before seeing any skills
PRIOR_SKILLS = 2         # how many "unknown" skills that assumption is worth

# Section headings, checked in this order ("Preferred qualifications" is preferred).
_PREFERRED_HEADING = re.compile(
    r"preferred|nice[\s-]to[\s-]have|bonus|desired|pluses", re.IGNORECASE)
_REQUIRED_HEADING = re.compile(
    r"qualifications|requirements|required|must[\s-]have|what you(?:'ll| will)? bring"
    r"|who you are|what we(?:'re| are) looking for|you have|skills", re.IGNORECASE)
_OTHER_HEADING = re.compile(
    r"responsibilities|what you(?:'ll| will)? do|about|benefits|overview|duties"
    r"|the role|the team|day[\s-]to[\s-]day|compensation|perks", re.IGNORECASE)
_MAX_HEADING_WORDS = 8
_BULLETS = ("•", "·", "-", "–", "* ", "o ")


@dataclass(frozen=True)
class MatchResult:
    score: int                      # 0-100
    skill_coverage: float | None    # 0-1, None if the posting names no known skills
    tfidf_similarity: float         # 0-1
    matched_skills: list[str]
    missing_skills: list[str]
    missing_required: list[str]     # the subset of missing_skills the posting requires
    confidence: str                 # "low" / "medium" / "high": evidence behind the score


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


def _heading_kind(line: str) -> str | None:
    """Return "required", "preferred" or "other" if the line is a section heading."""
    raw = line.strip()
    if raw.startswith(_BULLETS):  # "• Strong communication skills" is content, not a heading
        return None
    text = raw.strip("#*:").strip()
    if not text or len(text.split()) > _MAX_HEADING_WORDS or text.endswith("."):
        return None
    if _PREFERRED_HEADING.search(text):
        return "preferred"
    if _REQUIRED_HEADING.search(text):
        return "required"
    if _OTHER_HEADING.search(text):
        return "other"
    return None


def required_skills(posting_text: str) -> set[str]:
    """Skills named under required-type headings. Empty if the posting has no such heading."""
    found: set[str] = set()
    current: str | None = None
    for line in posting_text.splitlines():
        kind = _heading_kind(line)
        if kind is not None:
            current = kind
            continue
        if current == "required":
            found |= extract_skills(line)
    return found


def _coverage(matched: set[str], posting: set[str], required: set[str]) -> float:
    required = required & posting
    mentioned = posting - required
    if not required or not mentioned:  # no sections, or everything is required
        return len(matched) / len(posting)
    required_cov = len(matched & required) / len(required)
    mentioned_cov = len(matched & mentioned) / len(mentioned)
    return REQUIRED_WEIGHT * required_cov + MENTIONED_WEIGHT * mentioned_cov


def _adjusted_coverage(coverage: float, skill_count: int) -> float:
    """Pull coverage toward PRIOR_COVERAGE when the posting names few skills."""
    return (coverage * skill_count + PRIOR_COVERAGE * PRIOR_SKILLS) / (skill_count + PRIOR_SKILLS)


def _confidence(posting_skills: set[str], required: set[str]) -> str:
    """How much evidence the score rests on.

    Required skills carry 75% of the coverage, so when the posting has a
    requirements section, count only the known skills in it.
    """
    required = required & posting_skills
    skill_count = len(required) if required else len(posting_skills)
    if skill_count <= 2:
        return "low"
    if skill_count <= 5:
        return "medium"
    return "high"


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
    required = required_skills(posting_text)
    matched = posting_skills & resume_skills
    missing = posting_skills - resume_skills

    similarity = _tfidf_similarity(resume_text, posting_text)

    if posting_skills:
        coverage: float | None = _coverage(matched, posting_skills, required)
        adjusted = _adjusted_coverage(coverage, len(posting_skills))
        raw = COVERAGE_WEIGHT * adjusted + TFIDF_WEIGHT * similarity
    else:
        coverage = None
        raw = similarity

    return MatchResult(
        score=round(100 * min(max(raw, 0.0), 1.0)),
        skill_coverage=coverage,
        tfidf_similarity=similarity,
        matched_skills=sorted(matched),
        missing_skills=sorted(missing),
        missing_required=sorted(missing & required),
        confidence=_confidence(posting_skills, required),
    )
