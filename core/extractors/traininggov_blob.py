import re
import pandas as pd

def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

class TrainingGovBlobExtractor:
    name = "training.gov.au blob (Elements & Performance Criteria)"

    def can_handle(self, df: pd.DataFrame):
        cols = [c.lower() for c in df.columns]
        reasons = []
        score = 0

        blob_col = next((c for c in df.columns if "elements" in c.lower() and "performance" in c.lower()), None)
        if blob_col:
            score += 60
            reasons.append(f"Found blob column: {blob_col}")

        has_code = any(c in cols for c in ["unit code", "unit_code", "code"])
        has_title = any(c in cols for c in ["unit title", "unit_name", "title", "name"])
        if has_code:
            score += 20
            reasons.append("Detected Unit Code column")
        if has_title:
            score += 20
            reasons.append("Detected Unit Title column")

        if blob_col and df[blob_col].astype(str).head(3).str.contains(r"\b\d+\.\d+\b").any():
            score += 10
            reasons.append("Blob appears to contain performance criteria numbering (e.g., 1.1)")

        return min(score, 100), reasons

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        unit_code_col = next((c for c in df.columns if c.lower() in ["unit code", "unit_code", "code"]), None)
        unit_title_col = next((c for c in df.columns if c.lower() in ["unit title", "unit_name", "title", "name"]), None)
        blob_col = next((c for c in df.columns if "elements" in c.lower() and "performance" in c.lower()), None)

        if not (unit_code_col and unit_title_col and blob_col):
            raise ValueError("Missing required columns for TrainingGovBlobExtractor.")

        def extract_elements(unit_code: str, unit_title: str, blob: str):
            text = (blob or "").replace("\r\n", "\n").replace("\r", "\n")
            chunks = re.split(r"\n(?=(?:Element\s+\d+[:.]|\d+\.\s))", text, flags=re.I)
            rows = []

            for chunk in chunks:
                chunk = chunk.strip()
                if not chunk:
                    continue

                m = re.match(r"^(?:Element\s+\d+[:.]|\d+\.\s)\s*(.+)$", chunk, flags=re.I)
                element_title = _clean(m.group(1)) if m else None

                pcs = re.findall(r"(?m)^\s*(\d+\.\d+)\.?\s+(.*)$", chunk)
                pcs_text = "\n".join([f"{num}. {txt.strip()}" for num, txt in pcs]).strip() if pcs else ""

                if element_title:
                    rows.append({
                        "unit_code": unit_code,
                        "unit_title": unit_title,
                        "element_title": element_title,
                        "pcs_text": pcs_text
                    })
            return rows

        out_rows = []
        for _, r in df.iterrows():
            out_rows.extend(extract_elements(str(r[unit_code_col]), str(r[unit_title_col]), str(r[blob_col])))

        out = pd.DataFrame(out_rows).dropna(subset=["unit_code","unit_title","element_title"])
        out = out[out["element_title"].astype(str).str.len() > 0].reset_index(drop=True)
        return out
