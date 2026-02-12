# core/db.py
import os
import json
import hashlib
from datetime import datetime, timezone
from typing import Optional, Dict, Any

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


# ============================================================
# Helpers
# ============================================================

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_record_id(unit_code: str, element_title: str, pcs_text: str = "") -> str:
    """
    Stable deterministic id for an element record.
    Recommended: keep it stable across reruns of the same training package extraction.

    NOTE:
    - If you include pcs_text, any whitespace/format differences can change record_id.
    - If you want maximum stability, drop pcs_text from the hash.
    """
    raw = f"{(unit_code or '').strip()}||{(element_title or '').strip()}||{(pcs_text or '').strip()}".encode(
        "utf-8", errors="ignore"
    )
    return hashlib.sha256(raw).hexdigest()[:24]


def get_engine(database_url: Optional[str] = None) -> Engine:
    url = database_url or os.getenv("DATABASE_URL")
    if not url:
        raise ValueError("DATABASE_URL not set.")
    return create_engine(url, pool_pre_ping=True)


# ============================================================
# Schema / Migrations
# ============================================================

DDL_EXT = "CREATE EXTENSION IF NOT EXISTS pgcrypto;"

DDL = """
CREATE TABLE IF NOT EXISTS runs (
  run_id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at_utc     TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at_utc     TIMESTAMPTZ NOT NULL DEFAULT now(),
  status             TEXT NOT NULL DEFAULT 'running',

  source_filename    TEXT,
  source_fingerprint TEXT,
  training_package   TEXT,

  extractor_name     TEXT,
  extractor_version  TEXT,
  sil_version        TEXT,

  model              TEXT,
  settings_json      JSONB
);

-- IMPORTANT:
-- 1) record_id is NOT globally unique anymore.
-- 2) Primary key is (run_id, record_id) so each run can safely store its own records.
CREATE TABLE IF NOT EXISTS skill_records (
  run_id             UUID NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
  record_id          TEXT NOT NULL,

  created_at_utc      TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at_utc      TIMESTAMPTZ NOT NULL DEFAULT now(),

  row_index           INT,

  unit_code           TEXT,
  unit_title          TEXT,
  element_title       TEXT,
  pcs_text            TEXT,

  asced6_name         TEXT,

  skill_statement     TEXT,
  keywords_semicolon  TEXT,
  synonyms_semicolon  TEXT,

  qa_passes           BOOLEAN,
  qa_one_sentence     BOOLEAN,
  qa_word_count       INT,
  qa_has_method       BOOLEAN,
  qa_has_outcome      BOOLEAN,
  rewrite_count       INT,

  bart_model          TEXT,
  bart_temperature    FLOAT,
  bart_prompt         TEXT,

  sil_json            JSONB,

  PRIMARY KEY (run_id, record_id)
);

CREATE INDEX IF NOT EXISTS idx_skill_records_run_id ON skill_records(run_id);
CREATE INDEX IF NOT EXISTS idx_skill_records_unit_code ON skill_records(unit_code);
CREATE INDEX IF NOT EXISTS idx_skill_records_record_id ON skill_records(record_id);
CREATE INDEX IF NOT EXISTS idx_skill_records_run_row_index ON skill_records(run_id, row_index);
"""


def init_db(engine: Engine) -> None:
    """
    Creates base schema if missing AND applies a safe migration for older schemas.

    Migration addressed:
    - Older versions used record_id as PRIMARY KEY (global), causing rows to "move" between runs
      during upserts (and breaking batching).
    - We migrate to PRIMARY KEY (run_id, record_id).
    """
    with engine.begin() as conn:
        # Extension (ignore if not allowed)
        try:
            conn.execute(text(DDL_EXT))
        except Exception:
            pass

        # Create/ensure base tables
        for stmt in DDL.split(";"):
            s = stmt.strip()
            if s:
                conn.execute(text(s))

        # ---- Migration: if old schema exists (record_id as PK), fix it ----
        # Detect if skill_records has a primary key that is NOT (run_id, record_id)
        pk_cols = conn.execute(
            text("""
            SELECT kcu.column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON tc.constraint_name = kcu.constraint_name
             AND tc.table_schema = kcu.table_schema
            WHERE tc.table_schema = 'public'
              AND tc.table_name = 'skill_records'
              AND tc.constraint_type = 'PRIMARY KEY'
            ORDER BY kcu.ordinal_position
            """)
        ).fetchall()
        pk_cols = [r[0] for r in pk_cols] if pk_cols else []

        # If PK already correct, we're done.
        if pk_cols == ["run_id", "record_id"]:
            return

        # If table exists but PK is wrong (e.g., ["record_id"]), migrate.
        # 1) Add unique index on (run_id, record_id) first (non-destructive)
        conn.execute(text("""
        DO $$
        BEGIN
          IF NOT EXISTS (
            SELECT 1 FROM pg_indexes
            WHERE schemaname='public'
              AND tablename='skill_records'
              AND indexname='uq_skill_records_run_record'
          ) THEN
            CREATE UNIQUE INDEX uq_skill_records_run_record
            ON skill_records (run_id, record_id);
          END IF;
        END $$;
        """))

        # 2) Drop old PK constraint (name varies) if present and not correct
        if pk_cols and pk_cols != ["run_id", "record_id"]:
            # Find constraint name
            pk_name = conn.execute(
                text("""
                SELECT conname
                FROM pg_constraint
                WHERE conrelid='skill_records'::regclass
                  AND contype='p'
                """)
            ).scalar()
            if pk_name:
                conn.execute(text(f'ALTER TABLE skill_records DROP CONSTRAINT "{pk_name}";'))

        # 3) Add the correct composite PK (idempotent-ish)
        # If it already exists, this will error; wrap in DO block.
        conn.execute(text("""
        DO $$
        BEGIN
          IF NOT EXISTS (
            SELECT 1
            FROM pg_constraint
            WHERE conrelid='skill_records'::regclass
              AND contype='p'
          ) THEN
            ALTER TABLE skill_records
              ADD CONSTRAINT skill_records_pk PRIMARY KEY (run_id, record_id);
          END IF;
        END $$;
        """))


# ============================================================
# Run operations
# ============================================================

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


# ============================================================
# Record operations
# ============================================================

def upsert_skill_records(engine: Engine, run_id: str, batch_df: pd.DataFrame, *, row_index_start: int) -> None:
    """
    Upserts batch rows into skill_records.

    IMPORTANT FIX:
    - Conflict target is (run_id, record_id), not record_id alone.
      This prevents records being "moved" across runs, which breaks batching.
    """
    df = batch_df.copy()

    # Ensure record_id exists and is stable
    if "record_id" not in df.columns:
        df["record_id"] = [
            stable_record_id(str(u), str(e), str(p))
            for u, e, p in zip(df.get("unit_code", ""), df.get("element_title", ""), df.get("pcs_text", ""))
        ]

    # Persist row_index for resume cursor logic/debugging
    df["row_index"] = range(row_index_start, row_index_start + len(df))

    # Collect sil_json from extra columns
    base_cols = {
        "record_id", "row_index",
        "unit_code", "unit_title", "element_title", "pcs_text",
        "asced6_name",
        "skill_statement",
        "keywords", "keywords_semicolon", "synonyms_semicolon",
        "qa_passes", "qa_one_sentence", "qa_word_count", "qa_has_method", "qa_has_outcome", "rewrite_count",
        "bart_model", "bart_temperature", "bart_prompt",
    }

    sil_payloads = []
    for _, r in df.iterrows():
        extra = {}
        for c in df.columns:
            if c not in base_cols:
                v = r[c]
                if pd.isna(v):
                    continue
                extra[c] = v
        sil_payloads.append(extra if extra else None)

    df["sil_json"] = [json.dumps(x, ensure_ascii=False) if x else None for x in sil_payloads]

    # Normalize keyword naming
    if "keywords" in df.columns and "keywords_semicolon" not in df.columns:
        df["keywords_semicolon"] = df["keywords"]

    # Safe defaults
    if "bart_temperature" not in df.columns:
        df["bart_temperature"] = 0.2
    if "bart_model" not in df.columns:
        df["bart_model"] = None

    insert_sql = text("""
    INSERT INTO skill_records (
      run_id, record_id, row_index,
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
      :run_id, :record_id, :row_index,
      :unit_code, :unit_title, :element_title, :pcs_text,
      :asced6_name,
      :skill_statement,
      :keywords_semicolon, :synonyms_semicolon,
      :qa_passes, :qa_one_sentence, :qa_word_count, :qa_has_method, :qa_has_outcome, :rewrite_count,
      :bart_model, :bart_temperature, :bart_prompt,
      CAST(:sil_json AS JSONB),
      now(), now()
    )
    ON CONFLICT (run_id, record_id) DO UPDATE SET
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
    """
    Returns next row_index to process for this run, based on what's stored in DB.
    """
    with engine.begin() as conn:
        res = conn.execute(
            text("SELECT COALESCE(MAX(row_index), -1) AS mx FROM skill_records WHERE run_id=:run_id"),
            dict(run_id=run_id),
        ).mappings().first()
    mx = int(res["mx"]) if res and res["mx"] is not None else -1
    return mx + 1
