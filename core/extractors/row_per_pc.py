import re
import pandas as pd


def norm_col(c: str) -> str:
    c = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(c))  # split camelCase
    c = c.strip().lower()
    c = re.sub(r"[_\s]+", " ", c)  # collapse underscores/spaces
    return c


def find_unit_code_col(df: pd.DataFrame):
    for c in df.columns:
        n = norm_col(c)
        if n in {"unit code", "unitcode", "code", "national code", "nationalcode"}:
            return c
        if ("unit" in n and "code" in n) or ("national" in n and "code" in n):
            return c
    return None


def find_unit_title_col(df: pd.DataFrame):
    for c in df.columns:
        n = norm_col(c)
        if n in {"unit title", "unit name", "title", "name", "national title", "national name"}:
            return c
        if ("unit" in n and ("title" in n or "name" in n)) or ("national" in n and ("title" in n or "name" in n)):
            return c
    return None


def find_element_col(df: pd.DataFrame):
    for c in df.columns:
        n = norm_col(c)
        # must be an explicit element column, not "elements and performance criteria"
        if n == "element" or n.startswith("element "):
            return c
        if n in {"element title", "element name"}:
            return c
        if "element" in n and "performance" not in n and "criteria" not in n:
            return c
    return None


def find_pc_col(df: pd.DataFrame):
    for c in df.columns:
        n = norm_col(c)
        if n in {"performance criteria", "performance criterion", "criteria", "criterion", "pc"}:
            return c
        if "performance" in n and ("criteria" in n or "criterion" in n):
            return c
    return None


class RowPerPCExtractor:
    """
    For CSVs where each row is a single performance criterion and there is an explicit
    element column and explicit performance criteria column.
    """
    name = "row-per-performance-criteria"

    def can_handle(self, df: pd.DataFrame):
        reasons = []
        score = 0

        unit_code_col = find_unit_code_col(df)
        unit_title_col = find_unit_title_col(df)
        element_col = find_element_col(df)
        pc_col = find_pc_col(df)

        if unit_code_col:
            score += 25
            reasons.append(f"Found code column: {unit_code_col}")
        if unit_title_col:
            score += 25
            reasons.append(f"Found title column: {unit_title_col}")
        if element_col:
            score += 25
            reasons.append(f"Found element column: {element_col}")
        if pc_col:
            score += 25
            reasons.append(f"Found performance criteria column: {pc_col}")

        # Guardrail: do NOT match unless all required columns exist
        if not (unit_code_col and unit_title_col and element_col and pc_col):
            return 0, reasons

        return score, reasons

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        unit_code_col = find_unit_code_col(df)
        unit_title_col = find_unit_title_col(df)
        element_col = find_element_col(df)
        pc_col = find_pc_col(df)

        if not (unit_code_col and unit_title_col and element_col and pc_col):
            raise ValueError(
                "Missing required columns for RowPerPCExtractor. "
                f"Required: unit code, unit title, element, performance criteria. "
                f"Found columns: {list(df.columns)}"
            )

        rows = []
        grouped = df.groupby([unit_code_col, unit_title_col, element_col], dropna=False)
        for (uc, ut, el), g in grouped:
            pcs_text = "\n".join([str(x).strip() for x in g[pc_col].tolist() if str(x).strip()])
            rows.append({
                "unit_code": str(uc),
                "unit_title": str(ut),
                "element_title": str(el).strip(),
                "pcs_text": pcs_text.strip(),
            })

        out = pd.DataFrame(rows).dropna(subset=["unit_code","unit_title","element_title"]).reset_index(drop=True)
        return out
