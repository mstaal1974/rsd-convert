import re
import pandas as pd


# -----------------------------
# Helpers
# -----------------------------

def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def norm_col(c: str) -> str:
    """
    Normalise column names so we can match:
    - UnitCode / unit_code / Unit Code
    - National Code / NationalCode
    - ElementsAndPerformanceCriteria / Elements and Performance Criteria
    """
    c = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(c))  # split camelCase
    c = c.strip().lower()
    c = re.sub(r"[_\s]+", " ", c)  # collapse underscores/spaces
    return c


def find_unit_code_col(df: pd.DataFrame):
    for c in df.columns:
        n = norm_col(c)

        # Direct common names
        if n in {
            "unit code", "unitcode", "code",
            "national code", "nationalcode",
            "uoc code", "competency code", "unit id", "unitid"
        }:
            return c

        # Pattern matches
        if ("unit" in n and "code" in n) or ("national" in n and "code" in n):
            return c
        if ("uoc" in n and "code" in n) or ("competency" in n and "code" in n):
            return c

    return None


def find_unit_title_col(df: pd.DataFrame):
    for c in df.columns:
        n = norm_col(c)

        # Direct common names
        if n in {
            "unit title", "unit name", "title", "name",
            "national title", "national name",
            "uoc title", "uoc name",
            "competency title", "competency name"
        }:
            return c

        # Pattern matches
        if ("unit" in n and ("title" in n or "name" in n)):
            return c
        if ("national" in n and ("title" in n or "name" in n)):
            return c
        if ("uoc" in n and ("title" in n or "name" in n)):
            return c
        if ("competency" in n and ("title" in n or "name" in n)):
            return c

    return None


def find_blob_col(df: pd.DataFrame):
    """
    Find the column that contains the elements + performance criteria text.
    Examples seen in exports:
    - Elements and Performance Criteria
    - Elements & Performance Criteria
    - Elements and Criteria
    - Element and Performance Criteria (singular)
    """
    for c in df.columns:
        n = norm_col(c)

        # Strong signals
        if "element" in n and "performance" in n:
            return c
        if "elements" in n and "performance" in n:
            return c
        if "element" in n and "criteria" in n:
            return c
        if "elements" in n and "criteria" in n:
            return c

        # Sometimes abbreviated
        if "element" in n and "pc" in n:
            return c

    return None


# -----------------------------
# Extractor
# -----------------------------

class TrainingGovBlobExtractor:
    """
    Handles the "training.gov.au-style blob" where a single field includes:
    - Element headings
    - Performance criteria numbering (e.g., 1.1, 1.2...)
    """

    name = "training.gov.au blob (Elements & Performance Criteria)"

    def can_handle(self, df: pd.DataFrame):
        reasons = []
        score = 0

        unit_code_col = find_unit_code_col(df)
        unit_title_col = find_unit_title_col(df)
        blob_col = find_blob_col(df)

        if blob_col:
            score += 50
            reasons.append(f"Found elements/criteria blob column: {blob_col}")
        if unit_code_col:
            score += 25
            reasons.append(f"Found code column: {unit_code_col}")
        if unit_title_col:
            score += 25
            reasons.append(f"Found title column: {unit_title_col}")

        # Bonus if blob looks like it contains PC numbering (1.1 etc)
        if blob_col:
            try:
                sample = df[blob_col].astype(str).head(5)
                if sample.str.contains(r"\b\d+\.\d+\b").any():
                    score += 10
                    reasons.append("Blob appears to contain PC numbering (e.g., 1.1)")
            except Exception:
                pass

        # Critical guardrail: if any required column missing, DO NOT match
        if not (unit_code_col and unit_title_col and blob_col):
            return 0, reasons

        return min(score, 100), reasons

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        unit_code_col = find_unit_code_col(df)
        unit_title_col = find_unit_title_col(df)
        blob_col = find_blob_col(df)

        if not (unit_code_col and unit_title_col and blob_col):
            raise ValueError(
                "Missing required columns for TrainingGovBlobExtractor. "
                f"Required: (unit code, unit title, elements/criteria blob). "
                f"Found columns: {list(df.columns)}"
            )

        def extract_elements(unit_code: str, unit_title: str, blob: str):
            text = (blob or "").replace("\r\n", "\n").replace("\r", "\n")
            text = text.strip()
            if not text:
                return []

            # Split on likely element headings.
            # Supports: "Element 1: ...", "Element 2. ...", "1. ...", etc.
            chunks = re.split(r"\n(?=(?:Element\s+\d+[:.]|\d+\.\s))", text, flags=re.I)

            rows = []
            for chunk in chunks:
                chunk = chunk.strip()
                if not chunk:
                    continue

                # Identify element title line
                # Element 1: Title
                m1 = re.match(r"^(?:Element\s+\d+[:.])\s*(.+)$", chunk, flags=re.I)
                # 1. Title
                m2 = re.match(r"^(?:\d+\.)\s*(.+)$", chunk, flags=re.I)

                element_title = None
                if m1:
                    element_title = _clean(m1.group(1))
                elif m2:
                    element_title = _clean(m2.group(1))

                # Extract performance criteria lines like:
                # 1.1 Do X
                # 1.2 Do Y
                pcs = re.findall(r"(?m)^\s*(\d+\.\d+)\.?\s+(.*)$", chunk)
                pcs_text = "\n".join([f"{num}. {txt.strip()}" for num, txt in pcs]).strip() if pcs else ""

                # If we didn't reliably detect an element title, skip this chunk (avoid junk rows)
                if not element_title:
                    continue

                rows.append({
                    "unit_code": unit_code,
                    "unit_title": unit_title,
                    "element_title": element_title,
                    "pcs_text": pcs_text
                })

            return rows

        out_rows = []
        for _, r in df.iterrows():
            out_rows.extend(
                extract_elements(
                    str(r[unit_code_col]),
                    str(r[unit_title_col]),
                    str(r[blob_col])
                )
            )

        out = pd.DataFrame(out_rows).dropna(subset=["unit_code", "unit_title", "element_title"])
        out["element_title"] = out["element_title"].astype(str).map(_clean)
        out = out[out["element_title"].str.len() > 0].reset_index(drop=True)

        return out
