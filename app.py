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

# Optional module (you'll create this file per earlier message)
# If you haven't added it yet, keep the toggle OFF (default) or add the file.
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

def has_pc_token(s: str) -> bool:
    return bool(re.search(r"\b\d+\.\d+\b", str(s or "")))

with st.sidebar:
    st.header("Runtime")
    st.caption(f"OpenAI SDK version: {getattr(openai, '__version__', 'unknown')}")
    st.caption("Tip: keep Max elements low when testing.")

    st.divider()
    st.header("Settings")
    model = st.text_input("Model", value=default_model)
    max_rows = st.number_input("Max elements to process (cost control)", min_value=1, value=200)
    max_fixes = st.slider("Max auto-fixes per element (QA rewrite loop)", min_value=0, max_value=3, value=2)

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
    run_button = st.button("Run BART transformation", type="primary")

uploaded = st.file_uploader("Upload training package CSV", type=["csv"])

if not uploaded:
    st.info("Upload a training package CSV to begin.")
    st.stop()

# Read CSV
raw_df = pd.read_csv(uploaded)
st.subheader("Preview (raw)")
st.dataframe(raw_df.head(20), use_container_width=True)

# Extract / normalize
try:
    norm_df, extractor_used, scorecard = normalize_training_package_csv(raw_df, forced_extractor=forced)
    st.success(f"Extractor used: {extractor_used}")
except Exception as e:
    st.error(f"Extraction failed: {e}")
    st.stop()

if scorecard is not None:
    st.subheader("Extractor scorecard")
    st.dataframe(scorecard, use_container_width=True)

st.subheader("Normalized (Unit → Element → PCs)")
st.write(f"Detected **{len(norm_df)}** element records.")
st.dataframe(norm_df.head(30), use_container_width=True)

# Diagnostics: element titles polluted by PC tokens (should be 0)
if "element_title" in norm_df.columns:
    bad = norm_df[norm_df["element_title"].apply(has_pc_token)]
    if len(bad) > 0:
        st.warning(f"Diagnostics: {len(bad)} element titles contain PC tokens (e.g., 1.1). This should be 0.")
        st.dataframe(bad[["unit_code", "unit_title", "element_title"]].head(25), use_container_width=True)
    else:
        st.caption("Diagnostics: element titles are clean (no PC tokens detected).")

if not run_button:
    st.stop()

# OpenAI client
if not api_key:
    st.error("Missing OPENAI_API_KEY. Add it to .env (local) or Streamlit Secrets (cloud).")
    st.stop()

client = OpenAI(api_key=api_key)

work_df = norm_df.head(int(max_rows)).copy()
total = len(work_df)
if total == 0:
    st.error("No elements to process after normalization.")
    st.stop()

# Generate
skill_statements = []
prompts = []
qa_one_sentence = []
qa_word_count = []
qa_has_method = []
qa_has_outcome = []
qa_passes = []
keyword_list = []

progress = st.progress(0)
status = st.empty()

for i, (_, row) in enumerate(work_df.iterrows(), start=1):
    status.write(f"Generating {i}/{total} …")

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

    progress.progress(i / total)

work_df["skill_statement"] = skill_statements
work_df["bart_prompt"] = prompts
work_df["qa_one_sentence"] = qa_one_sentence
work_df["qa_word_count"] = qa_word_count
work_df["qa_has_method"] = qa_has_method
work_df["qa_has_outcome"] = qa_has_outcome
work_df["qa_passes"] = qa_passes

if generate_kw and KEYWORDS_AVAILABLE:
    work_df["keywords"] = keyword_list

status.write("Done ✅")

st.subheader("Results (sample)")
cols = ["unit_code", "unit_title", "element_title", "skill_statement", "qa_passes"]
if generate_kw and KEYWORDS_AVAILABLE:
    cols.append("keywords")
st.dataframe(work_df[cols].head(50), use_container_width=True)

# Export
rsd_df = to_rsd_rows(work_df)          # Ensure exporters.py maps Category & Keywords as you want
trace_df = to_traceability(work_df)

st.subheader("Downloads")
st.download_button(
    "Download RSD output CSV",
    data=rsd_df.to_csv(index=False).encode("utf-8"),
    file_name="rsd_output.csv",
    mime="text/csv",
)
st.download_button(
    "Download traceability CSV",
    data=trace_df.to_csv(index=False).encode("utf-8"),
    file_name="traceability.csv",
    mime="text/csv",
)
