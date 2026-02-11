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
def _read_excel_sheetnames(file) -> list:
    try:
        xls = pd.ExcelFile(file)
        return list(xls.sheet_names)
    except Exception:
        return []


def is_osca_structure_xlsx(file, filename: str) -> bool:
    """
    OSCA structure XLSX typically contains sheets named Table 1..5 and Table 5 is 'complete structure'.
    """
    name = (filename or "").lower()
    if not (name.endswith(".xlsx") or name.endswith(".xls")):
        return False
    sheets = _read_excel_sheetnames(file)
    s_norm = {str(s).strip().lower() for s in sheets}
    return ("table 5" in s_norm) or ("table 4" in s_norm and "contents" in s_norm)


def is_osca_index_titles(df: pd.DataFrame) -> bool:
    """
    Heuristic detection for the OSCA index of principal titles and alternative titles CSV.
    We look for columns that resemble principal/alternative titles.
    """
    cols = {_norm(c) for c in df.columns}
    has_principal = any("principal" in c and "title" in c for c in cols) or any(c in {"principal title"} for c in cols)
    has_alts = any("alternative" in c and "title" in c for c in cols) or any(c in {"alternative titles"} for c in cols)
    has_code = any("osca" in c and "code" in c for c in cols) or any(c in {"identifier", "code"} for c in cols)
    return bool(has_principal and (has_alts or has_code))


def load_osca_structure(
    engine: Engine,
    file,
    filename: str,
    *,
    taxonomy_version: Optional[str] = None,
    taxonomy_source: str = "official",
    sheet_name: Optional[str] = "Table 5",
) -> Dict[str, Any]:
    """
    Purpose-built loader for your OSCA structure XLSX.

    Your OSCA structure file (Table 5) is a "staircase" layout:

      col0: Major group code      (e.g., 1)
      col1: Major group name      (e.g., Managers)

      col1: Sub-major code        (e.g., 11)
      col2: Sub-major name

      col2: Minor group code      (e.g., 111)
      col3: Minor group name

      col3: Unit group code       (e.g., 1111)
      col4: Unit group name

      col4: Occupation code       (e.g., 111131)
      col5: Occupation title
      col6: Skill level (optional)

    We create records for:
      - major / sub_major / minor / unit_group / occupation
    in the `occupations` table, with parent_code relationships.
    """
    # IMPORTANT: ExcelFile reads the stream; reset pointer if needed
    try:
        file.seek(0)
    except Exception:
        pass

    sheets = _read_excel_sheetnames(file)
    if not sheets:
        raise ValueError("Could not read sheet names from XLSX")

    # Choose best sheet
    chosen = None
    if sheet_name and any(str(s).strip().lower() == str(sheet_name).strip().lower() for s in sheets):
        chosen = next(s for s in sheets if str(s).strip().lower() == str(sheet_name).strip().lower())
    else:
        # fallback to any table that looks like the complete structure
        for s in sheets:
            if str(s).strip().lower() == "table 5":
                chosen = s
                break
        if not chosen:
            chosen = sheets[0]

    # Read with header=None so we can find the actual start row
    try:
        file.seek(0)
    except Exception:
        pass

    raw = pd.read_excel(file, sheet_name=chosen, header=None)

    # Find the row where the real table begins.
    # In your file, row with "Identifier" appears in col0 a few rows above data.
    start_row = None
    for i in range(min(len(raw), 50)):
        row_vals = [str(x).strip().lower() for x in raw.iloc[i].tolist() if pd.notna(x)]
        if any(v == "identifier" for v in row_vals):
            start_row = i + 1
            break
    if start_row is None:
        # fallback: find the row containing "Occupation" (often near headers)
        for i in range(min(len(raw), 80)):
            row_vals = [str(x).strip().lower() for x in raw.iloc[i].tolist() if pd.notna(x)]
            if any(v == "occupation" for v in row_vals):
                start_row = i + 1
                break

    if start_row is None:
        raise ValueError("Could not locate the start of the OSCA structure table (Identifier row not found).")

    df = raw.iloc[start_row:].copy()
    # Keep only first 7 columns used by the structure
    df = df.iloc[:, :7]
    df.columns = ["major_code", "major_name",
                  "sub_code_or_name", "minor_code_or_name",
                  "unit_code_or_name", "occupation_name", "skill_level"]

    # The staircase columns mean codes and names are interleaved and shifted.
    # We'll reconstruct by reading the original raw columns by position.
    # Actual positions we observed:
    #   major_code in col0, major_name in col1
    #   sub_code in col1, sub_name in col2
    #   minor_code in col2, minor_name in col3
    #   unit_code in col3, unit_name in col4
    #   occ_code in col4, occ_name in col5
    #   skill_level in col6

    # Re-read the same region with numeric columns for safe indexing
    df2 = raw.iloc[start_row:, :7].copy()
    df2.columns = list(range(7))

    # Build canonical columns (codes + names)
    out = pd.DataFrame({
        "major_code": df2[0],
        "major_name": df2[1],
        "sub_code": df2[1],
        "sub_name": df2[2],
        "minor_code": df2[2],
        "minor_name": df2[3],
        "unit_code": df2[3],
        "unit_name": df2[4],
        "occ_code": df2[4],
        "occ_name": df2[5],
        "skill_level": df2[6],
    })

    # Forward fill group codes/names down the sheet
    for c in ["major_code", "major_name", "sub_code", "sub_name", "minor_code", "minor_name", "unit_code", "unit_name"]:
        out[c] = out[c].ffill()

    # Clean
    for c in out.columns:
        out[c] = out[c].apply(_to_str)

    # Helper to create occupation records
    records = {}
    def add_rec(level_type: str, code: str, name: str, parent_code: Optional[str]):
        if not code or not name:
            return
        occ_id = build_occupation_id("OSCA", code)
        if occ_id in records:
            return
        records[occ_id] = {
            "occupation_id": occ_id,
            "occupation_name": name,
            "taxonomy_source": taxonomy_source,
            "taxonomy_version": taxonomy_version,
            "major_group": None,
            "minor_group": None,
            "broad_group": None,
            "detailed_group": None,
            "description": None,
            "taxonomy_system": "OSCA",
            "taxonomy_code": code,
            "level_type": level_type,
            "parent_code": parent_code or None,
        }

    # Insert hierarchy levels + occupations
    for _, r in out.iterrows():
        major_code, major_name = r["major_code"], r["major_name"]
        sub_code, sub_name = r["sub_code"], r["sub_name"]
        minor_code, minor_name = r["minor_code"], r["minor_name"]
        unit_code, unit_name = r["unit_code"], r["unit_name"]
        occ_code, occ_name = r["occ_code"], r["occ_name"]

        # Heuristic: major codes are typically 1 digit; sub 2; minor 3; unit 4; occ 6
        # But we don’t rely on length — we rely on the staircase layout.
        add_rec("major", major_code, major_name, None)
        add_rec("sub_major", sub_code, sub_name, major_code)
        add_rec("minor", minor_code, minor_name, sub_code)
        add_rec("unit_group", unit_code, unit_name, minor_code)

        # Only add occupation when it looks like an occupation row (occ_code present AND occ_name present)
        if occ_code and occ_name:
            add_rec("occupation", occ_code, occ_name, unit_code)

    with engine.begin() as conn:
        conn.execute(UPSERT_OCC, list(records.values()))

    return {
        "status": "ok",
        "filename": filename,
        "sheet_used": chosen,
        "rows_read": len(out),
        "occupations_upserted": len(records),
        "note": "Loaded OSCA hierarchy (major/sub-major/minor/unit group/occupation) into occupations table."
    }


def load_osca_index_titles(
    engine: Engine,
    file,
    filename: str,
    *,
    taxonomy_version: Optional[str] = None,
    taxonomy_source: str = "official",
) -> Dict[str, Any]:
    """
    Loads OSCA 'Index of principal titles and alternative titles' (flat list).
    This is best used for synonyms/search enrichment.

    Minimum expected columns:
      - OSCA code / Identifier / Code
      - Principal Title
    Optional:
      - Alternative Titles (comma/semicolon separated)
    """
    df = _read_any(file, filename)

    code_col = _require(df, "osca_code", ["osca code", "identifier", "code", "taxonomy code", "occupation code"])
    principal_col = _require(df, "principal_title", ["principal title", "occupation name", "title", "name"])
    alt_col = _find_col(df, ["alternative titles", "alt titles", "alternate titles", "synonyms"])

    rows = []
    for _, r in df.iterrows():
        code = _to_str(r[code_col])
        name = _to_str(r[principal_col])
        if not code or not name:
            continue

        occ_id = build_occupation_id("OSCA", code)
        desc = None
        if alt_col:
            alts = _to_str(r[alt_col])
            if alts:
                desc = f"Alternative titles: {alts}"

        rows.append({
            "occupation_id": occ_id,
            "occupation_name": name,
            "taxonomy_source": taxonomy_source,
            "taxonomy_version": taxonomy_version,
            "major_group": None,
            "minor_group": None,
            "broad_group": None,
            "detailed_group": None,
            "description": desc,
            "taxonomy_system": "OSCA",
            "taxonomy_code": code,
            "level_type": "occupation",
            "parent_code": None,
        })

    with engine.begin() as conn:
        conn.execute(UPSERT_OCC, rows)

    return {"status": "ok", "filename": filename, "occupations_upserted": len(rows)}
