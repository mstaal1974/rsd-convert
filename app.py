# app.py
import os
import re
import json
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import openai

from core.extractor import normalize_training_package_csv, build_registry
from core.bart_generator import generate_skill_statement
from core.exporters import to_rsd_rows, to_traceability

# DB (Postgres/Neon/Supabase)
from core.db import (
    get_engine,
    init_db,
    create_run,
    upsert_skill_records,
    get_next_index,
    update_run_status,
    fetch_run_records,
)

# Optional keywords module
try:
    from core.keyword_generator import generate_keywords
    KEYWORDS_AVAILABLE = True
except Exception:
    KEYWORDS_AVAILABLE = False

load_dotenv()

st.set_page_config(page_title="Training Package → Element Skills (BART)", layout="wide")
st.title("Training Package → Element-level Skill Statements (BART)")

# -----------------------------
# Config
# -----------------------------
api_key = os.getenv("OPENAI_API_KEY", "") or st.secrets.get("OPENAI_API_KEY", "")
default_model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini") or st.secrets.get("OPENAI_MODEL", "gpt-4.1-mini")

# Prefer Streamlit Secrets for DB URL in cloud
db_url = os.getenv("DATABASE_URL", "") or st.secrets.get("DATABASE_URL", "")

reg = build_registry()


# -----------------------------
# Helpers
# -----------------------------
def has_pc_token(s: str) -> bool:
    return bool(re.search(r"\b\d+\.\d+\b", str(s or "")))


def fingerprint_df(df: pd.DataFrame) -> str:
    # stable-ish fingerprint to detect new uploads
    return f"{len(df)}|{','.join(map(str, df.columns))}"


def init_state():
    defaults = {
        "norm_df": None,
        "extractor_used": None,
        "scorecard": None,
        "last_file_fingerprint": None,

        # DB run tracking
        "run_id": None,
        "run_source_fingerprint": None,

        # UI
        "next_index_ui": 0,  # local pointer (DB is source of truth when resuming)
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# -----------------------------
# DB engine init
# -----------------------------
engine = None
db_ready = False
db_error = None

if db_url:
    try:
        engine = get_engine(db_url)
        init_db(engine)
        db_ready = True
    except Exception as e:
        db_error = str(e)
        db_ready = False


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Runtime")
    st.caption(f"OpenAI SDK version: {getattr(openai, '__version__', 'unknown')}")

    st.divider()
    st.header("Model + Cost Control")
    model = st.text_input("Model", value=default_model)
    max_fixes = st.slider("Max auto-fixes per element", min_value=0, max_value=3, value=1)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.1)

    st.divider()
    st.header("Batching / Resume")
    batch_size = st.number_input("Batch size (elements per run)", min_value=10, max_value=500, value=100, step=10)

    st.divider()
    st.header("Keywords")
    generate_kw = st.toggle(
        "Generate Keywords (AI)",
        value=False,
        disabled=not KEYWORDS_AVAILABLE,
        help="Adds a semicolon-separated keyword list per skill for search. Requires core/keyword_generator.py",
    )
    if not KEYWORDS_AVAILABLE:
        st.caption("Keywords module not found. Add core/keyword_generator.py to enable.")

    st.divider()
    st.header("Extractor")
    extractor_mode = st.radio("Mode", ["Auto-detect", "Choose"], index=0)
    forced = None
    if extractor_mode == "Choose":
        forced = st.selectbox("Extractor", reg.list_names())

    st.divider()
    st.header("Database")
    if not db_url:
        st.error("DATABASE_URL not set (Streamlit Secrets recommended). DB storage disabled.")
    elif not db_ready:
        st.error("DB connection failed. See error below.")
        st.code(db_error or "Unknown DB error")
    else:
        st.success("DB connected")

    resume_run_id = st.text_input("Resume an existing run_id (optional)", value="")
    use_resume = st.toggle("Resume from DB", value=False, disabled=not db_ready)

    st.divider()
    colA, colB = st.columns(2)
    run_batch = colA.button("Run next batch", type="primary", disabled=not api_key)
    reset_run = colB.button("Reset run (local)", disabled=False)

# Reset local run state (does NOT delete DB data)
if reset_run:
    st.session_state["run_id"] = None
    st.session_state["run_source_fingerprint"] = None
    st.session_state["next_index_ui"] = 0
    st.success("Local run state reset. (DB data unchanged)")

# -----------------------------
# Upload + Normalize
# -----------------------------
uploaded = st.file_uploader("Upload training package CSV", type=["csv"])
if not uploaded:
    st.info("Upload a training package CSV to begin.")
    st.stop()

raw_df = pd.read_csv(uploaded)
fp = fingerprint_df(raw_df)

# If new upload, reset cached normalization + run id (local)
if st.session_state["last_file_fingerprint"] != fp:
    st.session_state["last_file_fingerprint"] = fp
    st.session_state["norm_df"] = None
    st.session_state["extractor_used"] = None
    st.session_state["scorecard"] = None

    # If you want a new run per upload, reset local run id
    st.session_state["run_id"] = None
    st.session_state["run_source_fingerprint"] = fp
    st.session_state["next_index_ui"] = 0

st.subheader("Preview (raw)")
st.dataframe(raw_df.head(20), use_container_width=True)

# Normalize once per upload
if st.session_state["norm_df"] is None:
    try:
        norm_df, extractor_used, scorecard = normalize_training_package_csv(raw_df, forced_extractor=forced)
        st.session_state["norm_df"] = norm_df
        st.session_state["extractor_used"] = extractor_used
        st.session_state["scorecard"] = scorecard
    except Exception as e:
        st.error(f"Extraction failed: {e}")
        st.stop()

norm_df = st.session_state["norm_df"]
extractor_used = st.session_state["extractor_used"]
scorecard = st.session_state["scorecard"]

st.success(f"Extractor used: {extractor_used}")

if scorecard is not None:
    st.subheader("Extractor scorecard")
    st.dataframe(scorecard, use_container_width=True)

st.subheader("Normalized (Unit → Element → PCs)")
st.write(f"Detected **{len(norm_df)}** element records.")
st.dataframe(norm_df.head(30), use_container_width=True)

# Diagnostics: element titles should not contain PC tokens
if "element_title" in norm_df.columns:
    bad = norm_df[norm_df["element_title"].apply(has_pc_token)]
    if len(bad) > 0:
        st.warning(f"Diagnostics: {len(bad)} element titles contain PC tokens (e.g., 1.1). This should be 0.")
        st.dataframe(bad[["unit_code", "unit_title", "element_title"]].head(25), use_container_width=True)
    else:
        st.caption("Diagnostics: element titles are clean (no PC tokens detected).")

total = len(norm_df)
if total == 0:
    st.error("No elements found after normalization.")
    st.stop()

# -----------------------------
# Run selection (DB)
# -----------------------------
run_id = st.session_state.get("run_id")

if db_ready and use_resume and resume_run_id.strip():
    run_id = resume_run_id.strip()
    st.session_state["run_id"] = run_id

# Create a new run if none selected and DB is ready
if db_ready and not run_id:
    settings = {
        "batch_size": int(batch_size),
        "max_fixes": int(max_fixes),
        "temperature": float(temperature),
        "generate_keywords": bool(generate_kw),
        "extractor_mode": extractor_mode,
        "forced_extractor": forced,
    }
    try:
        run_id = create_run(
            engine,
            source_filename=getattr(uploaded, "name", "uploaded.csv"),
            source_fingerprint=fp,
            extractor_name=str(extractor_used),
            extractor_version="1.0.0",
            sil_version="1.0.0",
            model=model,
            settings=settings,
            training_package=None,
        )
        st.session_state["run_id"] = run_id
    except Exception as e:
        st.error(f"Failed to create DB run: {e}")
        st.stop()

# Display run id (or note DB disabled)
if db_ready and run_id:
    st.info(f"Run ID (save this to resume later): **{run_id}**")
elif not db_ready:
    st.warning("DB not available. This app will still run, but results won’t persist beyond the session.")

# Determine next index from DB if resuming, else from UI pointer
if db_ready and run_id:
    try:
        start_index = get_next_index(engine, run_id)
    except Exception as e:
        st.error(f"Failed to read resume position from DB: {e}")
        st.stop()
else:
    start_index = int(st.session_state["next_index_ui"])

end_index = min(total, start_index + int(batch_size))
st.info(f"Ready to process batch: rows **{start_index} → {end_index - 1}** (of {total}).")

# Show progress so far (DB)
if db_ready and run_id:
    try:
        processed_so_far = start_index
        st.write(f"Processed so far (DB): **{processed_so_far} / {total}**")
    except Exception:
        pass

# Show partial data (DB) + downloads (optional)
if db_ready and run_id:
    with st.expander("View stored results (DB) / partial downloads", expanded=False):
        db_df = fetch_run_records(engine, run_id)
        if len(db_df) == 0:
            st.caption("No stored rows yet for this run.")
        else:
            st.dataframe(
                db_df[["row_index", "unit_code", "unit_title", "element_title", "qa_passes"]].head(50),
                use_container_width=True,
            )
            # Build outputs from DB view
            rsd_partial = to_rsd_rows(db_df)
            trace_partial = to_traceability(db_df)
            st.download_button(
                "Download PARTIAL RSD output CSV (from DB)",
                data=rsd_partial.to_csv(index=False).encode("utf-8"),
                file_name="rsd_output_partial.csv",
                mime="text/csv",
            )
            st.download_button(
                "Download PARTIAL traceability CSV (from DB)",
                data=trace_partial.to_csv(index=False).encode("utf-8"),
                file_name="traceability_partial.csv",
                mime="text/csv",
            )

# -----------------------------
# Run batch
# -----------------------------
if not run_batch:
    st.stop()

if not api_key:
    st.error("Missing OPENAI_API_KEY. Add it to .env (local) or Streamlit Secrets (cloud).")
    st.stop()

client = OpenAI(api_key=api_key)

batch_df = norm_df.iloc[start_index:end_index].copy()
if len(batch_df) == 0:
    st.warning("No rows in this batch (you may already be finished).")
    if db_ready and run_id:
        update_run_status(engine, run_id, "completed")
    st.stop()

st.subheader("Running batch")
progress = st.progress(0)
status = st.empty()

# Generate for batch
skill_statements = []
prompts = []
qa_one_sentence = []
qa_word_count = []
qa_has_method = []
qa_has_outcome = []
qa_passes = []
rewrite_counts = []
keyword_list = []

for i, (_, row) in enumerate(batch_df.iterrows(), start=1):
    status.write(f"Generating {i}/{len(batch_df)} …")

    # Note: bart_generator currently owns its own temperature; if you want this control,
    # pass temperature through and use it in bart_generator.
    skill, qa, bart_prompt = generate_skill_statement(
        client=client,
        model=model,
        unit_code=row["unit_code"],
        unit_title=row["unit_title"],
        element_title=row["element_title"],
        pcs_text=row["pcs_text"],
        max_fixes=int(max_fixes),
    )

    skill_statements.append(skill)
    prompts.append(bart_prompt)
    qa_one_sentence.append(bool(qa.get("one_sentence")))
    qa_word_count.append(int(qa.get("word_count", 0)))
    qa_has_method.append(bool(qa.get("has_method_phrase")))
    qa_has_outcome.append(bool(qa.get("has_outcome_phrase")))
    qa_passes.append(bool(qa.get("passes")))
    rewrite_counts.append(int(qa.get("rewrite_count", 0)) if "rewrite_count" in qa else 0)

    if generate_kw and KEYWORDS_AVAILABLE:
        kws = generate_keywords(
            client=client,
            model=model,
            skill_statement=skill,
            pcs_text=row["pcs_text"],
        )
        keyword_list.append(kws)

    progress.progress(i / len(batch_df))

batch_df["skill_statement"] = skill_statements
batch_df["bart_prompt"] = prompts
batch_df["qa_one_sentence"] = qa_one_sentence
batch_df["qa_word_count"] = qa_word_count
batch_df["qa_has_method"] = qa_has_method
batch_df["qa_has_outcome"] = qa_has_outcome
batch_df["qa_passes"] = qa_passes
batch_df["rewrite_count"] = rewrite_counts
batch_df["bart_model"] = model
batch_df["bart_temperature"] = float(temperature)

if generate_kw and KEYWORDS_AVAILABLE:
    batch_df["keywords"] = keyword_list

status.write("Batch complete ✅")

# Persist batch to DB
if db_ready and run_id:
    try:
        upsert_skill_records(engine, run_id=run_id, batch_df=batch_df, row_index_start=start_index)
        update_run_status(engine, run_id, "running")
        st.success("Stored batch in DB ✅")
    except Exception as e:
        st.error(f"Failed to store batch in DB: {e}")
        st.stop()
else:
    # Local pointer update (non-persistent mode)
    st.session_state["next_index_ui"] = end_index
    st.warning("DB not available; results are not persisted.")

# Show batch results
st.subheader("Batch results (sample)")
cols = ["unit_code", "unit_title", "element_title", "skill_statement", "qa_passes"]
if generate_kw and KEYWORDS_AVAILABLE:
    cols.append("keywords")
st.dataframe(batch_df[cols].head(50), use_container_width=True)

# Finalize run if done
done = end_index >= total
if db_ready and run_id:
    update_run_status(engine, run_id, "completed" if done else "running")

st.subheader("Downloads (from DB)")
if db_ready and run_id:
    db_df = fetch_run_records(engine, run_id)
    if len(db_df) == 0:
        st.warning("No rows stored yet for this run.")
    else:
        rsd_df = to_rsd_rows(db_df)
        trace_df = to_traceability(db_df)

        st.download_button(
            "Download RSD output CSV (stored run)",
            data=rsd_df.to_csv(index=False).encode("utf-8"),
            file_name="rsd_output.csv" if done else "rsd_output_in_progress.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download traceability CSV (stored run)",
            data=trace_df.to_csv(index=False).encode("utf-8"),
            file_name="traceability.csv" if done else "traceability_in_progress.csv",
            mime="text/csv",
        )

        if done:
            st.success("All elements processed ✅")
        else:
            # show DB-derived next index
            nxt = get_next_index(engine, run_id)
            st.info(f"Next batch will start at index **{nxt}**")
else:
    st.caption("DB not configured; downloads are disabled in this mode. Configure DATABASE_URL to enable persistence.")
