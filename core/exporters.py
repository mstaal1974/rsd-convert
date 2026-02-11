import pandas as pd
from .templates import RSD_COLUMNS

def to_rsd_rows(norm_df: pd.DataFrame, author="training.gov.au") -> pd.DataFrame:
    rows = []
    for _, r in norm_df.iterrows():
        rows.append({
            "Canonical URL": "",
            "Unit of competency Name ": r["element_title"],
            "Author": author,
            "Skill Statement": r.get("skill_statement", ""),
            "Category": "",
            "Keywords": "",
            "Standards": "",
            "Certifications": f"{r['unit_code']} {r['unit_title']}",
            "Occupation Major Groups": "",
            "Occupation Minor Groups": "",
            "Broad Occupations": "",
            "Detailed Occupations": "",
            "O*Net Job Codes": "",
            "Employers": "",
            "Alignment Name": "",
            "Alignment URL": "",
        })
    return pd.DataFrame(rows, columns=RSD_COLUMNS)

def to_traceability(norm_df: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "unit_code","unit_title","element_title","pcs_text",
        "skill_statement","bart_prompt",
        "qa_one_sentence","qa_word_count","qa_has_method","qa_has_outcome","qa_passes"
    ]
    return norm_df[keep].copy()
