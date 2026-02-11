import pandas as pd
from core.extractor_registry import ExtractorRegistry

from core.extractors.traininggov_blob import TrainingGovBlobExtractor
from core.extractors.row_per_pc import RowPerPCExtractor

def build_registry() -> ExtractorRegistry:
    reg = ExtractorRegistry()
    reg.register(TrainingGovBlobExtractor())
    reg.register(RowPerPCExtractor())
    return reg

def normalize_training_package_csv(df: pd.DataFrame, forced_extractor: str | None = None):
    reg = build_registry()

    if forced_extractor:
        ex = reg.get(forced_extractor)
        norm = ex.extract(df)
        scorecard = None
        return norm, ex.name, scorecard

    ex, scorecard = reg.auto_select(df)
    norm = ex.extract(df)
    return norm, ex.name, scorecard
