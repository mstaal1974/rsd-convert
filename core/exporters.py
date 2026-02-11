import pandas as pd
from .templates import RSD_COLUMNS


# --------------------------------------------
# RSD OUTPUT
# --------------------------------------------
def to_rsd_rows(norm_df: pd.DataFrame, author="training.gov.au") -> pd.DataFrame:
    rows = []

    for _, r in norm_df.iterrows():
        rows.append({
            "Canonical URL": "",
            "Unit of competency Name ": r.get("element_title", ""),
            "Author": author,
            "Skill Statement": r.get("skill_statement", ""),
            # Inherit ASCED6 Name → Category
            "Category": r.get("asced6_name", ""),
            "Keywords": r.get("keywords", r.get("keywords_semicolon", "")),
            "Standards": "",
            "Certifications": f"{r.get('unit_code','')} {r.get('unit_title','')}",
            "Occupation Major Groups": "",
            "Occupation Minor Groups": "",
            "Broad Occupations": "",
            "Detailed Occupations": "",
            "O*Net Job Codes": r.get("onet_soc_codes", ""),
            "Employers": "",
            "Alignment Name": r.get("esco_skill_labels", ""),
            "Alignment URL": r.get("esco_skill_uris", ""),
        })

    return pd.DataFrame(rows, columns=RSD_COLUMNS)


# --------------------------------------------
# TRACEABILITY (Skill Intelligence Layer)
# --------------------------------------------
def to_traceability(norm_df: pd.DataFrame) -> pd.DataFrame:

    base_cols = [
        # Core identity
        "record_id",
        "run_id",
        "unit_code",
        "unit_title",
        "element_title",
        "pcs_text",
        "asced6_name",

        # Skill generation
        "skill_statement",
        "bart_model",
        "bart_temperature",
        "bart_prompt",

        # QA
        "qa_one_sentence",
        "qa_word_count",
        "qa_has_method",
        "qa_has_outcome",
        "qa_passes",
        "rewrite_count",

        # Search
        "keywords",
        "keywords_semicolon",
        "synonyms_semicolon",

        # Capability facets
        "capability_family",
        "capability_domain",
        "capability_subdomain",
        "capability_level",

        # Structure
        "primary_verb",
        "primary_object",

        # Alignment
        "onet_soc_codes",
        "onet_confidence",
        "esco_skill_uris",
        "esco_confidence",

        # Governance
        "requires_human_review",
        "confidence_score",

        # Full SIL payload
        "sil_json",
    ]

    # Only keep columns that actually exist
    existing = [c for c in base_cols if c in norm_df.columns]

    return norm_df[existing].copy()
