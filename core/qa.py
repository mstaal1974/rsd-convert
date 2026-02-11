import re

METHOD_RE = re.compile(r"\b(by|using|through|via)\b", re.I)
OUTCOME_RE = re.compile(r"\b(to ensure|to enable|to support|so that)\b", re.I)

def word_count(s: str) -> int:
    return len(re.findall(r"\b\w+\b", (s or "").strip()))

def one_sentence(s: str) -> bool:
    s = (s or "").strip()
    ends = re.findall(r"[.!?]", s)
    return len(ends) <= 1

def qa_check(skill_statement: str) -> dict:
    wc = word_count(skill_statement)
    return {
        "one_sentence": one_sentence(skill_statement),
        "word_count": wc,
        "has_method_phrase": bool(METHOD_RE.search(skill_statement or "")),
        "has_outcome_phrase": bool(OUTCOME_RE.search(skill_statement or "")),
        "passes": (
            one_sentence(skill_statement)
            and 25 <= wc <= 45
            and bool(METHOD_RE.search(skill_statement or ""))
            and bool(OUTCOME_RE.search(skill_statement or ""))
        ),
    }
