from __future__ import annotations
from dataclasses import dataclass
from typing import List, Protocol, Tuple
import pandas as pd

class Extractor(Protocol):
    name: str
    def can_handle(self, df: pd.DataFrame) -> Tuple[int, List[str]]:
        ...
    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        ...

@dataclass
class ExtractorEntry:
    name: str
    extractor: Extractor

class ExtractorRegistry:
    def __init__(self) -> None:
        self._entries: List[ExtractorEntry] = []

    def register(self, extractor: Extractor) -> None:
        self._entries.append(ExtractorEntry(name=extractor.name, extractor=extractor))

    def list_names(self) -> List[str]:
        return [e.name for e in self._entries]

    def get(self, name: str) -> Extractor:
        for e in self._entries:
            if e.name == name:
                return e.extractor
        raise KeyError(f"Extractor not found: {name}")

    def auto_select(self, df: pd.DataFrame):
        rows = []
        best = None
        best_score = -1

        for e in self._entries:
            score, reasons = e.extractor.can_handle(df)
            rows.append({
                "extractor": e.name,
                "score": score,
                "reasons": " | ".join(reasons) if reasons else ""
            })
            if score > best_score:
                best_score = score
                best = e.extractor

        scorecard = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)

        if best is None or best_score <= 0:
            raise ValueError("No extractor matched this CSV (all scored <= 0).")

        return best, scorecard
