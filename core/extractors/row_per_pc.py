import pandas as pd

class RowPerPCExtractor:
    name = "row-per-performance-criteria"

    def can_handle(self, df: pd.DataFrame):
        cols = [c.lower() for c in df.columns]
        score = 0
        reasons = []

        has_code = any(c in cols for c in ["unit code", "unit_code", "code"])
        has_title = any(c in cols for c in ["unit title", "unit_name", "title", "name"])
        has_element = any("element" in c for c in cols)
        has_pc = any(("performance" in c and "criteria" in c) for c in cols)

        if has_code: score += 20; reasons.append("Detected Unit Code column")
        if has_title: score += 20; reasons.append("Detected Unit Title column")
        if has_element: score += 30; reasons.append("Detected Element column")
        if has_pc: score += 30; reasons.append("Detected Performance Criteria column")

        return min(score, 100), reasons

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        unit_code_col = next((c for c in df.columns if c.lower() in ["unit code", "unit_code", "code"]), None)
        unit_title_col = next((c for c in df.columns if c.lower() in ["unit title", "unit_name", "title", "name"]), None)
        element_col = next((c for c in df.columns if "element" in c.lower()), None)
        pc_col = next((c for c in df.columns if ("performance" in c.lower() and "criteria" in c.lower())), None)

        if not (unit_code_col and unit_title_col and element_col and pc_col):
            raise ValueError("Missing required columns for RowPerPCExtractor.")

        rows = []
        grouped = df.groupby([unit_code_col, unit_title_col, element_col], dropna=False)
        for (uc, ut, el), g in grouped:
            pcs_text = "\n".join([str(x).strip() for x in g[pc_col].tolist() if str(x).strip()])
            rows.append({
                "unit_code": str(uc),
                "unit_title": str(ut),
                "element_title": str(el).strip(),
                "pcs_text": pcs_text.strip()
            })

        return pd.DataFrame(rows).reset_index(drop=True)
