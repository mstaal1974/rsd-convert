import re
import pandas as pd

# -----------------------------
# Output schema (normalized)
# -----------------------------
OUT_COLS = ["unit_code", "unit_title", "element_title", "pcs_text", "asced6_name"]


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
        if n in {
            "unit code", "unitcode", "code",
            "national code", "nationalcode",
            "uoc code", "competency code"
        }:
            return c
        if ("unit" in n and "code" in n) or ("national" in n and "code" in n) or ("uoc" in n and "code" in n) or (
            "competency" in n and "code" in n
        ):
            return c
    return None


def find_unit_title_col(df: pd.DataFrame):
    for c in df.columns:
        n = norm_col(c)
        if n in {
            "unit title", "unit name", "title", "name",
            "national title", "national name",
            "uoc title", "uoc name",
            "competency title", "competency name",
        }:
            return c
        if ("unit" in n and ("title" in n or "name" in n)) or ("national" in n and ("title" in n or "name" in n)) or (
            "uoc" in n and ("title" in n or "name" in n)
        ) or ("competency" in n and ("title" in n or "name" in n)):
            return c
    return None


def find_blob_col(df: pd.DataFrame):
    """
    Find the column that contains the elements + performance criteria text.
    Examples:
    - Elements and Performance Criteria
    - Elements & Performance Criteria
    - Elements and Criteria
    """
    for c in df.columns:
        n = norm_col(c)
        if "element" in n and ("performance" in n or "criteria" in n or "criterion" in n):
            return c
        if "elements" in n and ("performance" in n or "criteria" in n or "criterion" in n):
            return c
        if "element" in n and "pc" in n:
            return c
    return None


def find_asced6_name_col(df: pd.DataFrame):
    """
    Find ASCED6 Name column (used to populate Category in RSD output).
    Common headers:
    - ASCED6 Name
    - ASCED 6 Name
    """
    for c in df.columns:
        n = norm_col(c)
        if n in {"asced6 name", "asced 6 name", "asced name"}:
            return c
        if "asced6" in n and "name" in n:
            return c
    return None


def strip_pcs_from_element_title(title: str) -> str:
    """
    Safety net: truncate element title at first PC token (e.g., 1.1, 2.3)
    to prevent PC text leaking into element_title.
    """
    t = (title or "").strip()
    m = re.search(r"\b\d+\.\d+\b", t)
    if m:
        t = t[:m.start()].strip()
    return _clean(t)


def normalize_blob_text(text: str) -> str:
    """
    Many training.gov exports store the whole section as a single long string.
    We insert newlines before element headings and PC numbers.

    Handles both:
    - 1.1. Do something
    - 1.1 Do something
    """
    t = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"\s+", " ", t).strip()

    if not t:
        return ""

    # Insert newline before PCs like "1.1 Do..." OR "1.1. Do..."
    # and standardise to "1.1. " format
    t = re.sub(r"\s*(\d+\.\d+)\s*\.?\s+", r"\n\1. ", t)

    # Insert newline before element headings like "1. Title" but not "1.1"
    t = re.sub(r"\s*(\d+)\.(?!\d)\s+", r"\n\1. ", t)

    # Also support "Element 1:" style headings
    t = re.sub(r"\s*(Element\s+\d+)\s*[:.]\s*", r"\n\1: ", t, flags=re.I)

    lines = [ln.strip() for ln in t.split("\n") if ln.strip()]
    return "\n".join(lines)


# -----------------------------
# Extractor
# -----------------------------
class TrainingGovBlobExtractor:
    """
    Handles the "training.gov.au-style blob" where a single field includes:
    - Element headings (e.g., "1. Prepare ...", or "Element 1: Prepare ...")
    - Performance criteria (e.g., "1.1 ...", "1.2 ...")
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
            reasons.append(f"Found blob column: {blob_col}")
        if unit_code_col:
            score += 25
            reasons.append(f"Found code column: {unit_code_col}")
        if unit_title_col:
            score += 25
            reasons.append(f"Found title column: {unit_title_col}")

        # Bonus if blob sample contains PC numbering
        if blob_col:
            try:
                sample = df[blob_col].astype(str).head(5)
                if sample.str.contains(r"\b\d+\.\d+\b").any():
                    score += 10
                    reasons.append("Blob contains PC numbering (e.g., 1.1)")
            except Exception:
                pass

        # Guardrail: only match if ALL required columns exist
        if not (unit_code_col and unit_title_col and blob_col):
            return 0, reasons

        return min(score, 100), reasons

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        unit_code_col = find_unit_code_col(df)
        unit_title_col = find_unit_title_col(df)
        blob_col = find_blob_col(df)
        asced6_name_col = find_asced6_name_col(df)

        if not (unit_code_col and unit_title_col and blob_col):
            raise ValueError(
                "Missing required columns for TrainingGovBlobExtractor. "
                "Required: (unit code, unit title, elements/criteria blob). "
                f"Found columns: {list(df.columns)}"
            )

        def parse_element_blocks(blob_text: str):
            """
            Returns list of dicts: {"element_title": ..., "pcs_text": ...}
            """
            text = normalize_blob_text(blob_text)
            if not text:
                return []

            lines = text.split("\n")

            blocks = []
            current_element = None
            current_pcs = []

            # Patterns
            element_num_pat = re.compile(r"^(?P<num>\d+)\.\s+(?P<title>.+)$")
            element_word_pat = re.compile(r"^Element\s+\d+\s*:\s*(?P<title>.+)$", re.I)
            pc_pat = re.compile(r"^(?P<num>\d+\.\d+)\.\s+(?P<txt>.+)$")

            for ln in lines:
                lnl = ln.lower()

                # Ignore header / explanatory noise
                if lnl.startswith("elements performance criteria"):
                    continue
                if "elements describe the essential outcomes" in lnl:
                    continue
                if "performance criteria describe the performance needed" in lnl:
                    continue

                # PC line?
                m_pc = pc_pat.match(ln)
                if m_pc:
                    current_pcs.append(f"{m_pc.group('num')}. {m_pc.group('txt').strip()}")
                    continue

                # Element line?
                m_elw = element_word_pat.match(ln)
                m_eln = element_num_pat.match(ln)
                if m_elw or m_eln:
                    # flush previous block
                    if current_element:
                        blocks.append(
                            {
                                "element_title": current_element,
                                "pcs_text": "\n".join(current_pcs).strip(),
                            }
                        )
                    current_pcs = []

                    detected_title = (m_elw.group("title") if m_elw else m_eln.group("title"))
                    current_element = strip_pcs_from_element_title(detected_title)
                    continue

            # flush last block
            if current_element:
                blocks.append(
                    {
                        "element_title": current_element,
                        "pcs_text": "\n".join(current_pcs).strip(),
                    }
                )

            return blocks

        out_rows = []
        for _, r in df.iterrows():
            unit_code = str(r[unit_code_col])
            unit_title = str(r[unit_title_col])
            blob = str(r[blob_col])

            asced6_name = ""
            if asced6_name_col:
                asced6_name = str(r[asced6_name_col]).strip()

            blocks = parse_element_blocks(blob)
            for b in blocks:
                out_rows.append(
                    {
                        "unit_code": unit_code,
                        "unit_title": unit_title,
                        "element_title": b["element_title"],
                        "pcs_text": b["pcs_text"],
                        "asced6_name": asced6_name,
                    }
                )

        # Always return expected schema
        if not out_rows:
            return pd.DataFrame(columns=OUT_COLS)

        out = pd.DataFrame(out_rows, columns=OUT_COLS)
        out["element_title"] = out["element_title"].astype(str).map(strip_pcs_from_element_title)
        out = out.dropna(subset=["unit_code", "unit_title", "element_title"])
        out = out[out["element_title"].astype(str).str.len() > 0].reset_index(drop=True)
        return out
