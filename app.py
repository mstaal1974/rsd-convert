import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from core.extractor import normalize_training_package_csv, build_registry
from core.bart_generator import generate_skill_statement
from core.exporters import to_rsd_rows, to_traceability

load_dotenv()

st.set_page_config(page_title="Training Package → Element Skills (BART)", layout="wide")
st.title("Training Package → Element-level Skill Statements (BART)")

api_key = os.getenv("OPENAI_API_KEY", "")
default_model = os.getenv("OPENAI_MODEL", "gpt-5.2")

reg = build_registry()

with st.sidebar:
    st.header("Settings")
    model = st.text_input("Model", value=default_model)
    max_rows = st.number_input("Max elements to process (testing / cost control)", min_value=1, value=200)
    max_fixes = st.slider("Max auto-fixes per element (QA rewrite loop)", min_value=0, max_value=3, value=2)

    st.divider()
    st.header("Extractor")
    extractor_mode = st.radio("Mode", ["Auto-detect", "Choose"], index=0)
    forced = None
    if extractor_mode == "Choose":
        forced = st.selectbox("Extractor", reg.list_names())

    run_button = st.button("Run BART transformation", type="primary")

uploaded = st.file_uploader("Upload training package CSV", type=["csv"])

if uploaded:
    raw_df = pd.read_csv(uploaded)
    st.subheader("Preview (raw)")
    st.dataframe(raw_df.head(20), use_container_width=True)

    try:
        norm_df, extractor_used, scorecard = normalize_training_package_csv(raw_df, forced_extractor=forced)
        st.success(f"Extractor used: {extractor_used}")

        if scorecard is not None:
            st.subheader("Extractor scorecard")
            st.dataframe(scorecard, use_container_width=True)

        st.subheader("Normalized (Unit → Element → PCs)")
        st.write(f"Detected **{len(norm_df)}** element records.")
        st.dataframe(norm_df.head(20), use_container_width=True)

    except Exception as e:
        st.error(f"Extraction failed: {e}")
        st.stop()

    if run_button:
        if not api_key:
            st.error("Missing OPENAI_API_KEY. Add it to .env (local) or Streamlit Secrets (cloud).")
            st.stop()

        client = OpenAI(api_key=api_key)

        work_df = norm_df.head(int(max_rows)).copy()

        skill_statements = []
        prompts = []
        qa_one_sentence = []
        qa_word_count = []
        qa_has_method = []
        qa_has_outcome = []
        qa_passes = []

        progress = st.progress(0)
        total = len(work_df)

        for _, row in work_df.iterrows():
            skill, qa, prompt = generate_skill_statement(
                client=client,
                model=model,
                unit_code=row["unit_code"],
                unit_title=row["unit_title"],
                element_title=row["element_title"],
                pcs_text=row["pcs_text"],
                max_fixes=int(max_fixes),
            )
            skill_statements.append(skill)
            prompts.append(prompt)

            qa_one_sentence.append(qa["one_sentence"])
            qa_word_count.append(qa["word_count"])
            qa_has_method.append(qa["has_method_phrase"])
            qa_has_outcome.append(qa["has_outcome_phrase"])
            qa_passes.append(qa["passes"])

            progress.progress(min(1.0, (len(skill_statements) / total)))

        work_df["skill_statement"] = skill_statements
        work_df["bart_prompt"] = prompts
        work_df["qa_one_sentence"] = qa_one_sentence
        work_df["qa_word_count"] = qa_word_count
        work_df["qa_has_method"] = qa_has_method
        work_df["qa_has_outcome"] = qa_has_outcome
        work_df["qa_passes"] = qa_passes

        st.subheader("Results (sample)")
        st.dataframe(
            work_df[["unit_code","unit_title","element_title","skill_statement","qa_passes"]].head(50),
            use_container_width=True
        )

        rsd_df = to_rsd_rows(work_df)
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
