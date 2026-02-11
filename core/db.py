import os
import json
import hashlib
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Iterable

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


# ---------- helpers ----------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_record_id(unit_code: str, element_title: str, pcs_text: str) -> str:
    """
    Stable, deterministic id for an element record, to support idempotent upserts.
    """
    raw = f"{unit_code}||{element_title}||{pcs_text}".encode("utf-8", errors="ignore")
    return hashlib.sha256(raw).hexdigest()[:24]


def get_engine(database_url: Optional[str] = None) -> Engine:
    url = database_url or os.getenv("DATABASE_URL")
    if not url:
        raise ValueError("DATABASE_URL not set.")
    return create_engine(url, pool_pre_ping=True)


# ---------- schema ----------

DDL = """
CREATE TABLE IF NOT EXISTS runs (
  run_id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at_utc    TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at_utc    TIMESTAMPTZ NOT NULL DEFAULT now(),
  status            TEXT NOT NULL DEFAULT 'running',

  source_filename   TEXT,
  source_fingerprint TEXT,
  training_package  TEXT,

  extractor_name    TEXT,
  extractor_version TEXT,
  sil_version       TEXT,

  model             TEXT,
  settings_json     JSONB
);

CREATE TABLE IF NOT EXISTS skill_records (
  record_id         TEXT PRIMARY KEY,
  run_id            UUID NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,

  created_at_utc    TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at_utc    TIMESTAMPTZ NOT NULL DEFAULT now(),

  row_index         INT,

  unit_code         TEXT,
  unit_title        TEXT,
  element_title     TEXT,
  pcs_text          TEXT,

  asced6_name       TEXT,

  skill_statement   TEXT,
  keywords_semicolon TEXT,
  synonyms_semicolon TEXT,

  qa_passes         BOOLEAN,
  qa_one_sentence   BOOLEAN,
  qa_word_count     INT,
  qa_has_method     BOOLEAN,
  qa_has_outcome    BOOLEAN,
  rewrite_count     INT,

  bart_model        TEXT,
  bart_temperature  FLOAT,
  bart_prompt       TEXT,

  sil_json          JSONB
);

CREATE INDEX IF NOT EXISTS idx_skill_records_run_id ON skill_records(run_id);
CREATE INDEX IF NOT EXISTS idx_skill_records_unit_code ON skill_records(unit_code);
"""

# Note: gen_random_uuid() requires pgcrypto in Postgres.
# Supabase has it; on Neon you may need:
#   CREATE EXTENSION IF NOT EXISTS pgcrypto;

DDL_EXT = "CREATE EXTENSION IF NOT EXISTS pgcrypto;"


def init_db(engine: Engine) -> None:
    with engine.begin() as conn:
        try:
            conn.execute(text(DDL_EXT))
        except Exception:
            # Some managed Postgres setups may not allow extensions; ignore if already available
            pass
        for stmt in DDL.split(";"):
            s = stmt.strip()
            if s:
                conn.execute(text(s))


# ---------- run ops ----------

def create_run(
    engine: Engine,
    *,
    source_filename: str,
    source_fingerprint: str,
    extractor_name: str,
    extractor_version: str = "unknown",
    sil_version: str = "1.0.0",
    model: str,
    settings: Dict[str, Any],
    training_package: Optional[str] = None,
) -> str:
    with engine.begin() as conn:
        res = conn.execute(
            text("""
            INSERT INTO runs (
              source_filename, source_fingerprint, training_package,
              extractor_name, extractor_version, sil_version,
              model, settings_json, status, updated_at_utc
            )
            VALUES (
              :source_filename, :source_fingerprint, :training_package,
              :extractor_name, :extractor_version, :sil_version,
              :model, :settings_json, 'running', now()
            )
            RETURNING run_id
            """),
            dict(
                source_filename=source_filename,
                source_fingerprint=source_fingerprint,
                training_package=training_package,
                extractor_name=extractor_name,
                extractor_version=extractor_version,
                sil_version=sil_version,
                model=model,
                settings_json=json.dumps(settings),
            ),
        )
        return str(res.scalar())


def update_run_status(engine: Engine, run_id: str, status: str) -> None:
    with engine.begin() as conn:
        conn.execute(
            text("""
            UPDATE runs
            SET status=:status, updated_at_utc=now()
            WHERE run_id=:run_id
            """),
            dict(status=status, run_id=run_id),
        )


def get_run(engine: Engine, run_id: str) -> Optional[Dict[str, Any]]:
    with engine.begin() as conn:
        res = conn.execute(
            text("SELECT * FROM runs WHERE run_id=:run_id"),
            dict(run_id=run_id),
        ).mappings().first()
        return dict(res) if res else None


# ---------- record ops ----------

def upsert_skill_records(engine: Engine, run_id: str, batch_df: pd.DataFrame, *, row_index_start: int) -> None:
    """
    batch_df: your processed batch with skill_statement + QA + (optional) keywords etc.
    stores both standard columns + a flexible sil_json blob for the rest.
    """
    # ensure stable record_id
    df = batch_df.copy()

    if "record_id" not in df.columns:
        df["record_id"] = [
            stable_record_id(str(u), str(e), str(p))
            for u, e, p in zip(df["unit_code"], df["element_title"], df["pcs_text"])
        ]

    # row_index persisted for resume debugging
    df["row_index"] = range(row_index_start, row_index_start + len(df))

    # collect sil_json from whatever extra fields exist
    base_cols = {
        "record_id","row_index",
        "unit_code","unit_title","element_title","pcs_text",
        "asced6_name",
        "skill_statement",
        "keywords","keywords_semicolon","synonyms_semicolon",
        "qa_passes","qa_one_sentence","qa_word_count","qa_has_method","qa_has_outcome","rewrite_count",
        "bart_model","bart_temperature","bart_prompt",
    }

    sil_payloads = []
    for _, r in df.iterrows():
        extra = {}
        for c in df.columns:
            if c not in base_cols:
                v = r[c]
                # avoid NaN issues
                if pd.isna(v):
                    continue
                extra[c] = v
        sil_payloads.append(extra if extra else None)

    df["sil_json"] = [json.dumps(x, ensure_ascii=False) if x else None for x in sil_payloads]

    # normalize keyword column naming
    if "keywords" in df.columns and "keywords_semicolon" not in df.columns:
        df["keywords_semicolon"] = df["keywords"]

    # safe defaults
    if "bart_temperature" not in df.columns:
        df["bart_temperature"] = 0.2
    if "bart_model" not in df.columns:
        df["bart_model"] = None

    insert_sql = text("""
    INSERT INTO skill_records (
      record_id, run_id, row_index,
      unit_code, unit_title, element_title, pcs_text,
      asced6_name,
      skill_statement,
      keywords_semicolon, synonyms_semicolon,
      qa_passes, qa_one_sentence, qa_word_count, qa_has_method, qa_has_outcome, rewrite_count,
      bart_model, bart_temperature, bart_prompt,
      sil_json,
      created_at_utc, updated_at_utc
    )
    VALUES (
      :record_id, :run_id, :row_index,
      :unit_code, :unit_title, :element_title, :pcs_text,
      :asced6_name,
      :skill_statement,
      :keywords_semicolon, :synonyms_semicolon,
      :qa_passes, :qa_one_sentence, :qa_word_count, :qa_has_method, :qa_has_outcome, :rewrite_count,
      :bart_model, :bart_temperature, :bart_prompt,
      CAST(:sil_json AS JSONB),
      now(), now()
    )
    ON CONFLICT (record_id) DO UPDATE SET
      run_id = EXCLUDED.run_id,
      row_index = EXCLUDED.row_index,
      unit_code = EXCLUDED.unit_code,
      unit_title = EXCLUDED.unit_title,
      element_title = EXCLUDED.element_title,
      pcs_text = EXCLUDED.pcs_text,
      asced6_name = EXCLUDED.asced6_name,
      skill_statement = EXCLUDED.skill_statement,
      keywords_semicolon = EXCLUDED.keywords_semicolon,
      synonyms_semicolon = EXCLUDED.synonyms_semicolon,
      qa_passes = EXCLUDED.qa_passes,
      qa_one_sentence = EXCLUDED.qa_one_sentence,
      qa_word_count = EXCLUDED.qa_word_count,
      qa_has_method = EXCLUDED.qa_has_method,
      qa_has_outcome = EXCLUDED.qa_has_outcome,
      rewrite_count = EXCLUDED.rewrite_count,
      bart_model = EXCLUDED.bart_model,
      bart_temperature = EXCLUDED.bart_temperature,
      bart_prompt = EXCLUDED.bart_prompt,
      sil_json = EXCLUDED.sil_json,
      updated_at_utc = now()
    """)

    records = df.to_dict(orient="records")
    for rec in records:
        # ensure all keys exist
        rec.setdefault("asced6_name", "")
        rec.setdefault("keywords_semicolon", "")
        rec.setdefault("synonyms_semicolon", "")
        rec.setdefault("rewrite_count", 0)
        rec["run_id"] = run_id

    with engine.begin() as conn:
        conn.execute(insert_sql, records)


def fetch_run_records(engine: Engine, run_id: str) -> pd.DataFrame:
    with engine.begin() as conn:
        res = conn.execute(
            text("""
            SELECT *
            FROM skill_records
            WHERE run_id=:run_id
            ORDER BY row_index ASC
            """),
            dict(run_id=run_id),
        ).mappings().all()
    return pd.DataFrame([dict(r) for r in res]) if res else pd.DataFrame()


def get_next_index(engine: Engine, run_id: str) -> int:
    with engine.begin() as conn:
        res = conn.execute(
            text("SELECT COALESCE(MAX(row_index), -1) AS mx FROM skill_records WHERE run_id=:run_id"),
            dict(run_id=run_id),
        ).mappings().first()
    mx = int(res["mx"]) if res and res["mx"] is not None else -1
    return mx + 1
