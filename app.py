import os
import re
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import openai

from core.extractor import normalize_training_package_csv, build_registry
from core.bart_generator import generate_skill_statement
from core.exporters import to_rsd_rows, to_traceability

# Optional keywords module
try:
    from core.keyword_generator import generate_keywords
    KEYWORDS_AVAILABLE = True
except Exception:
    KEYWORDS_AVAILABLE = False

load_dotenv()

st.set_page_config(page_title="Training Package → Element Skills (BART)", layout="wide")
st.title("Training Package → Element-level Skill Statements (BART)")

api_key = os.getenv("OPENAI_API_KEY", "")
default_model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

reg = build_registry()


# -----------------------------
# Helpers
# -----------------------------
def has_pc_token(s: str) -> bool:
    return bool(re.search(r"\b\d+\.\d+\b", str(s or "")))


def init_state():
    for k, v in {
        "norm_df": None,
        "extractor_used": None,
        "scorecard": None,
        "results_df": None,  # accumulated processed rows
        "next_index": 0,     # resume pointer
        "last_file_fingerprint": None,
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v


def fingerprint_df(df: pd.DataFrame) -> str:
    # stable-ish fingerprint to detect new uploads
    return f"{len(df)}|{','.join(map(str, df.columns))}"


init_state()

# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("Runtime")
    st.caption(f"OpenAI SDK version: {getattr(openai, '__version__', 'unknown')}")

    st.divider()
    st.header("Model + Cost Control")
    model = st.text_input("Model", value=default_model)
    max_fixes = st.slider("Max auto-fixes per element", min_value=0, max_value=3, value=1)

    st.caption("For large runs: set fixes to 0–1 and use batching.")

    st.divider()
    st.header("Batching / Resume")
    batch_size = st.number_input("Batch size (elements per run)", min_value=10, max_value=500, value=100, step=10)
    manual_start = st.number_input("Start index override (optional)", min_value=0, value=0, step=1)
    use_manual_start = st.toggle("Use start index override", value=False)

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
    colA, colB = st.columns(2)
    run_batch = colA.button("Run next batch", type="primary")
    reset_run = colB.button("Reset run")

# Reset run state (user-triggered)
if reset_run:
    st.session_state["results_df"] = None
    st.session_state["next_index"] = 0
    st.success("Run state reset. Upload a file and start again.")

uploaded = st.file_uploader("Upload training package CSV", type=["csv"])

if not uploaded:
    st.info("Upload a training package CSV to begin.")
    st.stop()

raw_df = pd.read_csv(uploaded)

# Detect new upload and reset state if needed
fp = fingerprint_df(raw_df)
if st.session_state["last_file_fingerprint"] != fp:
    st.session_state["last_file_fingerprint"] = fp
    st.session_state["results_df"] = None
    st.session_state["next_index"] = 0
    st.session_state["norm_df"] = None
    st.session_state["scorecard"] = None
    st.session_state["extractor_used"] = None

st.subheader("Preview (raw)")
st.dataframe(raw_df.head(20), use_container_width=True)

# Normalize / extract (only once per upload)
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

# Diagnostics: element titles should NOT contain PC tokens
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

# Determine start index (resume)
start_index = int(manual_start) if use_manual_start else int(st.session_state["next_index"])
start_index = max(0, min(start_index, total))  # clamp
end_index = min(total, start_index + int(batch_size))

st.info(f"Ready to process batch: rows **{start_index} → {end_index - 1}** (of {total}).")

# Show accumulated progress
processed_so_far = 0 if st.session_state["results_df"] is None else len(st.session_state["results_df"])
st.write(f"Processed so far: **{processed_so_far} / {total}**")

# Allow download of partial results anytime
if st.session_state["results_df"] is not None and len(st.session_state["results_df"]) > 0:
    st.subheader("Partial downloads (so far)")
    partial_rsd = to_rsd_rows(st.session_state["results_df"])
    partial_trace = to_traceability(st.session_state["results_df"])
    st.download_button(
        "Download PARTIAL RSD output CSV",
        data=partial_rsd.to_csv(index=False).encode("utf-8"),
        file_name="rsd_output_partial.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download PARTIAL traceability CSV",
        data=partial_trace.to_csv(index=False).encode("utf-8"),
        file_name="traceability_partial.csv",
        mime="text/csv",
    )

# Only run generation when user clicks
if not run_batch:
    st.stop()

# OpenAI client
if not api_key:
    st.error("Missing OPENAI_API_KEY. Add it to .env (local) or Streamlit Secrets (cloud).")
    st.stop()

client = OpenAI(api_key=api_key)

batch_df = norm_df.iloc[start_index:end_index].copy()
if len(batch_df) == 0:
    st.warning("No rows in this batch (you may already be finished).")
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
keyword_list = []

for i, (_, row) in enumerate(batch_df.iterrows(), start=1):
    status.write(f"Generating {i}/{len(batch_df)} …")

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
    qa_one_sentence.append(qa["one_sentence"])
    qa_word_count.append(qa["word_count"])
    qa_has_method.append(qa["has_method_phrase"])
    qa_has_outcome.append(qa["has_outcome_phrase"])
    qa_passes.append(qa["passes"])

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

if generate_kw and KEYWORDS_AVAILABLE:
    batch_df["keywords"] = keyword_list

status.write("Batch complete ✅")

# Accumulate results
if st.session_state["results_df"] is None:
    st.session_state["results_df"] = batch_df
else:
    st.session_state["results_df"] = pd.concat([st.session_state["results_df"], batch_df], ignore_index=True)

# Update resume pointer
st.session_state["next_index"] = end_index

# Show batch results
st.subheader("Batch results (sample)")
cols = ["unit_code", "unit_title", "element_title", "skill_statement", "qa_passes"]
if generate_kw and KEYWORDS_AVAILABLE:
    cols.append("keywords")
st.dataframe(batch_df[cols].head(50), use_container_width=True)

# Final downloads if complete
done = st.session_state["next_index"] >= total
st.subheader("Downloads")
rsd_df = to_rsd_rows(st.session_state["results_df"])
trace_df = to_traceability(st.session_state["results_df"])

st.download_button(
    "Download RSD output CSV (current)",
    data=rsd_df.to_csv(index=False).encode("utf-8"),
    file_name="rsd_output.csv" if done else "rsd_output_in_progress.csv",
    mime="text/csv",
)
st.download_button(
    "Download traceability CSV (current)",
    data=trace_df.to_csv(index=False).encode("utf-8"),
    file_name="traceability.csv" if done else "traceability_in_progress.csv",
    mime="text/csv",
)

if done:
    st.success("All elements processed ✅")
else:
    st.info(f"Next batch will start at index **{st.session_state['next_index']}**")
