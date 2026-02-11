import os
import pandas as pd
import streamlit as st
from sqlalchemy import text
from openai import OpenAI

from core.db import get_engine, init_db
from core.etl_loaders import (
    load_qualifications_and_units,
    load_occupation_taxonomy,
    load_occupation_qualification_map,
    load_occupation_crosswalk,
    is_osca_structure_xlsx,
    is_osca_index_titles,
    load_osca_structure,
    load_osca_index_titles,
)

from core.occupation_profiles import build_occupation_skill_profile, build_all_occupation_profiles

st.set_page_config(page_title="Admin • DB + Taxonomy + Profiles", layout="wide")
st.title("Admin • DB + Taxonomy + Occupation Skill Profiles")

db_url = os.getenv("DATABASE_URL", "") or st.secrets.get("DATABASE_URL", "")
if not db_url:
    st.error("DATABASE_URL not set in Streamlit Secrets.")
    st.stop()

engine = get_engine(db_url)
init_db(engine)

# -----------------------------
# Quick DB health
# -----------------------------
with st.expander("DB Health Check", expanded=True):
    try:
        with engine.begin() as conn:
            one = conn.execute(text("SELECT 1")).scalar()
        st.success(f"DB OK (SELECT 1 → {one})")
    except Exception as e:
        st.error("DB connection failed")
        st.code(str(e))
        st.stop()

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(["Load Qualifications/Units", "Load Taxonomy + Mapping", "Build/Query Profiles"]) 


# -----------------------------
# Tab 1: Qualifications + Units
# -----------------------------
with tab1:
    st.subheader("Load Qualifications + Qualification→Units")
    st.caption("Upload your Qualifications_*.csv (or equivalent). This populates `qualifications` and `qualification_units`.")

    q_file = st.file_uploader("Upload Qualifications/Units file (CSV or XLSX)", type=["csv", "xlsx"], key="qual_file")

    colA, colB, colC = st.columns(3)
    tp = colA.text_input("Training package (optional)", value="MSL")
    release = colB.text_input("Release (optional)", value="")
    source_file = colC.text_input("Source label (optional)", value="upload")

    if q_file:
        try:
    df_preview = pd.read_csv(q_file) if q_file.name.lower().endswith(".csv") else pd.read_excel(q_file)
except ImportError as e:
    st.error("Reading Excel requires openpyxl. Add `openpyxl` to requirements.txt and redeploy.")
    st.code(str(e))
    st.stop()

        st.dataframe(df_preview.head(25), use_container_width=True)

        if st.button("Load Qualifications + Units into DB", type="primary"):
            q_file.seek(0)
            res = load_qualifications_and_units(
                engine,
                q_file,
                q_file.name,
                training_package=tp or None,
                release=release or None,
                source_file=source_file or q_file.name,
            )
            st.success(res)


# -----------------------------
# Tab 2: Taxonomy + Mapping
# -----------------------------
with tab2:
    st.subheader("Load ANZSCO / OSCA taxonomy and mappings")

st.markdown("### Load taxonomy (auto-detect OSCA structure vs OSCA index)")
t_ver = st.text_input("Taxonomy version (optional)", value="")

t_file = st.file_uploader("Upload taxonomy file (CSV or XLSX)", type=["csv", "xlsx"], key="tax_file")

if t_file:
    name = t_file.name.lower()

    # Peek safely (don’t consume file permanently)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        st.info("Detected Excel. Checking if this is OSCA Structure…")
        t_file.seek(0)
        is_structure = is_osca_structure_xlsx(t_file, t_file.name)

        if is_structure:
            st.success("Auto-detected: OSCA Structure (hierarchy) XLSX")
            if st.button("Load OSCA Structure into DB", type="primary"):
                t_file.seek(0)
                res = load_osca_structure(
                    engine,
                    t_file,
                    t_file.name,
                    taxonomy_version=t_ver or None,
                    taxonomy_source="official",
                    sheet_name="Table 5",
                )
                st.success(res)
        else:
            st.warning("Excel file does not look like OSCA Structure. Loading as generic taxonomy.")
            t_sys = st.selectbox("Taxonomy system", ["ANZSCO", "OSCA"], index=1)
            t_file.seek(0)
            tdf = pd.read_excel(t_file)
            st.dataframe(tdf.head(25), use_container_width=True)

            if st.button("Load taxonomy into DB"):
                t_file.seek(0)
                res = load_occupation_taxonomy(
                    engine,
                    t_file,
                    t_file.name,
                    taxonomy_system=t_sys,
                    taxonomy_version=t_ver or None,
                    taxonomy_source="official",
                )
                st.success(res)

    else:
        # CSV: could be OSCA index titles
        t_file.seek(0)
        tdf = pd.read_csv(t_file)
        st.dataframe(tdf.head(25), use_container_width=True)

        if is_osca_index_titles(tdf):
            st.success("Auto-detected: OSCA Index of Principal/Alternative Titles (flat list)")
            if st.button("Load OSCA Index Titles into DB", type="primary"):
                t_file.seek(0)
                res = load_osca_index_titles(
                    engine,
                    t_file,
                    t_file.name,
                    taxonomy_version=t_ver or None,
                    taxonomy_source="official",
                )
                st.success(res)
        else:
            st.warning("CSV does not match OSCA index heuristic. Loading as generic taxonomy.")
            t_sys = st.selectbox("Taxonomy system", ["ANZSCO", "OSCA"], index=0)
            if st.button("Load taxonomy into DB"):
                t_file.seek(0)
                res = load_occupation_taxonomy(
                    engine,
                    t_file,
                    t_file.name,
                    taxonomy_system=t_sys,
                    taxonomy_version=t_ver or None,
                    taxonomy_source="official",
                )
                st.success(res)


    st.divider()
    st.markdown("### Load occupation → qualification map")
    oq_file = st.file_uploader("Upload occupation→qualification map (CSV or XLSX)", type=["csv", "xlsx"], key="oq_file")
    st.caption("Required columns: taxonomy_system, taxonomy_code, qualification_code. Optional: confidence_score, mapping_method, notes.")

    if oq_file:
        oqdf = pd.read_csv(oq_file) if oq_file.name.lower().endswith(".csv") else pd.read_excel(oq_file)
        st.dataframe(oqdf.head(25), use_container_width=True)

        if st.button("Load occupation→qualification map into DB", type="primary"):
            oq_file.seek(0)
            res = load_occupation_qualification_map(engine, oq_file, oq_file.name, default_mapping_method="manual")
            st.success(res)

    st.divider()
    st.markdown("### Load ANZSCO ↔ OSCA crosswalk (optional)")
    cw_file = st.file_uploader("Upload crosswalk file (CSV or XLSX)", type=["csv", "xlsx"], key="cw_file")
    st.caption("Required columns: from_taxonomy_system, from_taxonomy_code, to_taxonomy_system, to_taxonomy_code.")

    if cw_file:
        cwdf = pd.read_csv(cw_file) if cw_file.name.lower().endswith(".csv") else pd.read_excel(cw_file)
        st.dataframe(cwdf.head(25), use_container_width=True)

        if st.button("Load crosswalk into DB"):
            cw_file.seek(0)
            res = load_occupation_crosswalk(engine, cw_file, cw_file.name, default_mapping_type="official")
            st.success(res)


# -----------------------------
# Tab 3: Build / Query / Export
# -----------------------------
with tab3:
    st.subheader("Build occupation skill profiles")
    st.caption("Profiles are built by inheritance: occupation → qualifications → units → skill_records (element skills).")

    # Choose run_id filter (optional)
    run_id = st.text_input("Optional: constrain to run_id (leave blank to use all skill_records)", value="")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Build one occupation")
        occ_id = st.text_input("occupation_id (e.g., ANZSCO:311411)", value="")
        replace = st.toggle("Replace existing profile rows for this occupation", value=True)

        if st.button("Build profile for occupation", type="primary"):
            if not occ_id.strip():
                st.error("Enter an occupation_id")
            else:
                res = build_occupation_skill_profile(
                    engine,
                    occ_id.strip(),
                    run_id=run_id.strip() or None,
                    replace=replace,
                )
                st.success(res)

    with col2:
        st.markdown("### Build all mapped occupations")
        replace_all = st.toggle("Replace existing profile rows for ALL occupations", value=False)

        if st.button("Build profiles for all occupations"):
            res = build_all_occupation_profiles(
                engine,
                run_id=run_id.strip() or None,
                replace=replace_all,
            )
            st.success({"status": res["status"], "count": res["count"]})

    st.divider()
    st.subheader("Query / Export occupation skill profiles")

    q_occ = st.text_input("Filter by occupation_id (optional)", value="")
    q_limit = st.number_input("Limit", min_value=50, max_value=20000, value=2000, step=50)

    if st.button("Run query"):
        sql = """
        SELECT
          osp.occupation_id,
          o.occupation_name,
          osp.qualification_code,
          q.qualification_title,
          osp.unit_code,
          osp.unit_type,
          osp.weight,
          sr.element_title,
          sr.skill_statement,
          sr.keywords_semicolon,
          sr.asced6_name,
          osp.record_id
        FROM occupation_skill_profiles osp
        LEFT JOIN occupations o ON o.occupation_id = osp.occupation_id
        LEFT JOIN qualifications q ON q.qualification_code = osp.qualification_code
        LEFT JOIN skill_records sr ON sr.record_id = osp.record_id
        WHERE 1=1
        """
        params = {}

        if q_occ.strip():
            sql += " AND osp.occupation_id = :occ"
            params["occ"] = q_occ.strip()

        sql += " ORDER BY osp.occupation_id, osp.qualification_code, osp.unit_code LIMIT :lim"
        params["lim"] = int(q_limit)

        with engine.begin() as conn:
            rows = conn.execute(text(sql), params).mappings().all()

        df = pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()
        st.write(f"Returned {len(df)} rows")
        st.dataframe(df.head(200), use_container_width=True)

        st.download_button(
            "Download occupation_skill_profiles.csv",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="occupation_skill_profiles.csv",
            mime="text/csv",
        )
