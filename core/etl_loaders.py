import re
import json
import pandas as pd
from typing import Optional, Dict, Any, Tuple, List
from sqlalchemy import text
from sqlalchemy.engine import Engine


# -------------------------
# Column matching helpers
# -------------------------
def _norm(s: str) -> str:
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(s))
    s = s.strip().lower()
    s = re.sub(r"[_\s]+", " ", s)
    return s


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cand = set(candidates)
    for c in df.columns:
        if _norm(c) in cand:
            return c
    return None


def _require(df: pd.DataFrame, key: str, candidates: List[str]) -> str:
    c = _find_col(df, candidates)
    if not c:
        raise ValueError(f"Missing required column for {key}. Tried: {candidates}. Found: {list(df.columns)}")
    return c


def _to_str(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return str(x).strip()


def _read_any(file, filename: str) -> pd.DataFrame:
    name = (filename or "").lower()
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(file)
    return pd.read_csv(file)


def build_occupation_id(taxonomy_system: str, taxonomy_code: str) -> str:
    return f"{_to_str(taxonomy_system).upper()}:{_to_str(taxonomy_code)}"


# -------------------------
# Upsert SQL
# -------------------------
UPSERT_QUAL = text("""
INSERT INTO qualifications (
  qualification_code, qualification_title, training_package, release, asced6_name,
  created_at_utc, updated_at_utc
)
VALUES (
  :qualification_code, :qualification_title, :training_package, :release, :asced6_name,
  now(), now()
)
ON CONFLICT (qualification_code) DO UPDATE SET
  qualification_title = EXCLUDED.qualification_title,
  training_package = EXCLUDED.training_package,
  release = EXCLUDED.release,
  asced6_name = EXCLUDED.asced6_name,
  updated_at_utc = now()
""")

UPSERT_QUAL_UNIT = text("""
INSERT INTO qualification_units (
  qualification_code, unit_code, unit_title, unit_type, group_name, source_file,
  created_at_utc, updated_at_utc
)
VALUES (
  :qualification_code, :unit_code, :unit_title, :unit_type, :group_name, :source_file,
  now(), now()
)
ON CONFLICT (qualification_code, unit_code) DO UPDATE SET
  unit_title = EXCLUDED.unit_title,
  unit_type = EXCLUDED.unit_type,
  group_name = EXCLUDED.group_name,
  source_file = EXCLUDED.source_file,
  updated_at_utc = now()
""")

UPSERT_OCC = text("""
INSERT INTO occupations (
  occupation_id, occupation_name, taxonomy_source, taxonomy_version,
  major_group, minor_group, broad_group, detailed_group, description,
  taxonomy_system, taxonomy_code, level_type, parent_code,
  created_at_utc, updated_at_utc
)
VALUES (
  :occupation_id, :occupation_name, :taxonomy_source, :taxonomy_version,
  :major_group, :minor_group, :broad_group, :detailed_group, :description,
  :taxonomy_system, :taxonomy_code, :level_type, :parent_code,
  now(), now()
)
ON CONFLICT (occupation_id) DO UPDATE SET
  occupation_name = EXCLUDED.occupation_name,
  taxonomy_source = EXCLUDED.taxonomy_source,
  taxonomy_version = EXCLUDED.taxonomy_version,
  major_group = EXCLUDED.major_group,
  minor_group = EXCLUDED.minor_group,
  broad_group = EXCLUDED.broad_group,
  detailed_group = EXCLUDED.detailed_group,
  description = EXCLUDED.description,
  taxonomy_system = EXCLUDED.taxonomy_system,
  taxonomy_code = EXCLUDED.taxonomy_code,
  level_type = EXCLUDED.level_type,
  parent_code = EXCLUDED.parent_code,
  updated_at_utc = now()
""")

UPSERT_OCC_QUAL = text("""
INSERT INTO occupation_qualifications (
  occupation_id, qualification_code, mapping_method, confidence_score, notes,
  created_at_utc, updated_at_utc
)
VALUES (
  :occupation_id, :qualification_code, :mapping_method, :confidence_score, :notes,
  now(), now()
)
ON CONFLICT (occupation_id, qualification_code) DO UPDATE SET
  mapping_method = EXCLUDED.mapping_method,
  confidence_score = EXCLUDED.confidence_score,
  notes = EXCLUDED.notes,
  updated_at_utc = now()
""")

UPSERT_CROSSWALK = text("""
INSERT INTO occupation_crosswalk (
  from_taxonomy_system, from_taxonomy_code,
  to_taxonomy_system, to_taxonomy_code,
  mapping_type, confidence_score, notes,
  created_at_utc, updated_at_utc
)
VALUES (
  :from_taxonomy_system, :from_taxonomy_code,
  :to_taxonomy_system, :to_taxonomy_code,
  :mapping_type, :confidence_score, :notes,
  now(), now()
)
ON CONFLICT (from_taxonomy_system, from_taxonomy_code, to_taxonomy_system, to_taxonomy_code) DO UPDATE SET
  mapping_type = EXCLUDED.mapping_type,
  confidence_score = EXCLUDED.confidence_score,
  notes = EXCLUDED.notes,
  updated_at_utc = now()
""")


# -------------------------
# Loaders
# -------------------------
def load_qualifications_and_units(
    engine: Engine,
    file,
    filename: str,
    *,
    training_package: Optional[str] = None,
    release: Optional[str] = None,
    source_file: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Loads:
      - qualifications (qualification_code/title, package/release/asced6_name if present)
      - qualification_units (qualification_code + unit_code + core/elective grouping)

    Works with your uploaded Qualifications_MSL_*.csv AND similar packaging exports.
    """
    df = _read_any(file, filename)

    # Common packaging columns (flexible)
    qual_code = _require(df, "qualification_code", [
        "qualification code", "qualificationcode", "qual code", "code", "national code"
    ])
    qual_title = _find_col(df, ["qualification title", "qualification name", "title", "name", "national title"]) or qual_code

    unit_code = _require(df, "unit_code", [
        "unit code", "unitcode", "uoc code", "national unit code", "unit national code"
    ])
    unit_title = _find_col(df, ["unit title", "unit name", "uoc title", "national unit title"])

    unit_type = _find_col(df, ["unit type", "core elective", "core/elective", "core or elective", "type"])
    group_name = _find_col(df, ["group", "group name", "elective group", "grouping"])

    asced6 = _find_col(df, ["asced6 name", "asced 6 name", "asced name"])
    tp_col = _find_col(df, ["training package", "trainingpackage", "package"])
    rel_col = _find_col(df, ["release", "version"])

    q_rows = {}
    qu_rows = []

    for _, r in df.iterrows():
        qc = _to_str(r[qual_code])
        if not qc:
            continue

        qt = _to_str(r[qual_title]) if qual_title else ""
        uc = _to_str(r[unit_code])
        if not uc:
            continue

        ut = _to_str(r[unit_title]) if unit_title else ""
        utype = _to_str(r[unit_type]).lower() if unit_type else "core"
        if utype in ("c", "core units", "core unit"):
            utype = "core"
        if utype in ("e", "elective units", "elective unit"):
            utype = "elective"
        if utype == "":
            utype = "core"

        grp = _to_str(r[group_name]) if group_name else ""

        q_rows[qc] = {
            "qualification_code": qc,
            "qualification_title": qt or qc,
            "training_package": training_package or (_to_str(r[tp_col]) if tp_col else None),
            "release": release or (_to_str(r[rel_col]) if rel_col else None),
            "asced6_name": _to_str(r[asced6]) if asced6 else None,
        }

        qu_rows.append({
            "qualification_code": qc,
            "unit_code": uc,
            "unit_title": ut or None,
            "unit_type": utype,
            "group_name": grp or None,
            "source_file": source_file or filename,
        })

    with engine.begin() as conn:
        conn.execute(UPSERT_QUAL, list(q_rows.values()))
        conn.execute(UPSERT_QUAL_UNIT, qu_rows)

    return {
        "status": "ok",
        "qualifications_upserted": len(q_rows),
        "qualification_units_upserted": len(qu_rows),
        "filename": filename,
    }


def load_occupation_taxonomy(
    engine: Engine,
    file,
    filename: str,
    *,
    taxonomy_system: str,           # 'ANZSCO' or 'OSCA'
    taxonomy_version: Optional[str] = None,
    taxonomy_source: str = "official",
) -> Dict[str, Any]:
    """
    Loads ANZSCO/OSCA taxonomy rows. Your file must contain at minimum:
      - taxonomy_code (e.g. ANZSCO code / OSCA code)
      - occupation_name (label)
    Optional:
      - level_type, parent_code, description, groups
    """
    df = _read_any(file, filename)

    code_col = _require(df, "taxonomy_code", ["taxonomy code", "code", "anzsco code", "osca code", "occupation code"])
    name_col = _require(df, "occupation_name", ["occupation name", "name", "title", "label", "occupation title"])

    level_col = _find_col(df, ["level type", "level", "type"])
    parent_col = _find_col(df, ["parent code", "parent", "parent taxonomy code"])

    desc_col = _find_col(df, ["description", "desc"])
    major_col = _find_col(df, ["major group", "major"])
    minor_col = _find_col(df, ["minor group", "minor"])
    broad_col = _find_col(df, ["broad group", "broad"])
    detailed_col = _find_col(df, ["detailed group", "detailed"])

    rows = []
    for _, r in df.iterrows():
        code = _to_str(r[code_col])
        name = _to_str(r[name_col])
        if not code or not name:
            continue

        occ_id = build_occupation_id(taxonomy_system, code)

        rows.append({
            "occupation_id": occ_id,
            "occupation_name": name,
            "taxonomy_source": taxonomy_source,
            "taxonomy_version": taxonomy_version,
            "major_group": _to_str(r[major_col]) if major_col else None,
            "minor_group": _to_str(r[minor_col]) if minor_col else None,
            "broad_group": _to_str(r[broad_col]) if broad_col else None,
            "detailed_group": _to_str(r[detailed_col]) if detailed_col else None,
            "description": _to_str(r[desc_col]) if desc_col else None,
            "taxonomy_system": taxonomy_system.upper(),
            "taxonomy_code": code,
            "level_type": _to_str(r[level_col]).lower() if level_col else None,
            "parent_code": _to_str(r[parent_col]) if parent_col else None,
        })

    with engine.begin() as conn:
        conn.execute(UPSERT_OCC, rows)

    return {"status": "ok", "occupations_upserted": len(rows), "filename": filename}


def load_occupation_qualification_map(
    engine: Engine,
    file,
    filename: str,
    *,
    default_mapping_method: str = "manual",
) -> Dict[str, Any]:
    """
    Expects a file with at minimum:
      - taxonomy_system (ANZSCO/OSCA)
      - taxonomy_code
      - qualification_code
    Optional:
      - confidence_score, notes, mapping_method
    """
    df = _read_any(file, filename)

    sys_col = _require(df, "taxonomy_system", ["taxonomy system", "system", "taxonomy"])
    code_col = _require(df, "taxonomy_code", ["taxonomy code", "code", "occupation code", "anzsco code", "osca code"])
    qual_col = _require(df, "qualification_code", ["qualification code", "qualification", "qual code"])

    conf_col = _find_col(df, ["confidence", "confidence score", "score"])
    notes_col = _find_col(df, ["notes", "note"])
    method_col = _find_col(df, ["mapping method", "method"])

    rows = []
    for _, r in df.iterrows():
        sys = _to_str(r[sys_col]).upper()
        code = _to_str(r[code_col])
        q = _to_str(r[qual_col])
        if not sys or not code or not q:
            continue

        rows.append({
            "occupation_id": build_occupation_id(sys, code),
            "qualification_code": q,
            "mapping_method": _to_str(r[method_col]) or default_mapping_method if method_col else default_mapping_method,
            "confidence_score": float(r[conf_col]) if conf_col and _to_str(r[conf_col]) else None,
            "notes": _to_str(r[notes_col]) if notes_col else None,
        })

    with engine.begin() as conn:
        conn.execute(UPSERT_OCC_QUAL, rows)

    return {"status": "ok", "occupation_qualifications_upserted": len(rows), "filename": filename}


def load_occupation_crosswalk(
    engine: Engine,
    file,
    filename: str,
    *,
    default_mapping_type: str = "official",
) -> Dict[str, Any]:
    """
    Expects:
      - from_taxonomy_system, from_taxonomy_code
      - to_taxonomy_system, to_taxonomy_code
    Optional: mapping_type, confidence_score, notes
    """
    df = _read_any(file, filename)

    f_sys = _require(df, "from_taxonomy_system", ["from taxonomy system", "from system", "from taxonomy"])
    f_code = _require(df, "from_taxonomy_code", ["from taxonomy code", "from code"])
    t_sys = _require(df, "to_taxonomy_system", ["to taxonomy system", "to system", "to taxonomy"])
    t_code = _require(df, "to_taxonomy_code", ["to taxonomy code", "to code"])

    type_col = _find_col(df, ["mapping type", "type"])
    conf_col = _find_col(df, ["confidence", "confidence score", "score"])
    notes_col = _find_col(df, ["notes", "note"])

    rows = []
    for _, r in df.iterrows():
        rows.append({
            "from_taxonomy_system": _to_str(r[f_sys]).upper(),
            "from_taxonomy_code": _to_str(r[f_code]),
            "to_taxonomy_system": _to_str(r[t_sys]).upper(),
            "to_taxonomy_code": _to_str(r[t_code]),
            "mapping_type": _to_str(r[type_col]) or default_mapping_type if type_col else default_mapping_type,
            "confidence_score": float(r[conf_col]) if conf_col and _to_str(r[conf_col]) else None,
            "notes": _to_str(r[notes_col]) if notes_col else None,
        })

    with engine.begin() as conn:
        conn.execute(UPSERT_CROSSWALK, rows)

    return {"status": "ok", "crosswalk_rows_upserted": len(rows), "filename": filename}
