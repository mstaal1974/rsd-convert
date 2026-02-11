# pages/90_Admin_DB.py
import os
import streamlit as st
import pandas as pd
from sqlalchemy import text

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

# -----------------------------
# DB Setup
# -----------------------------
db_url = os.getenv("DATABASE_URL", "") or st.secrets.get("DATABASE_URL", "")
if not db_url:
    st.error("DATABASE_URL not set in Streamlit Secrets.")
    st.stop()

try:
    engine = get_engine(db_url)
    init_db(engine)
except Exception as e:
    st.error("DB init failed")
    st.code(str(e))
    st.stop()

# -----------------------------
# Helpers
# -----------------------------
def safe_read_dataframe(uploaded_file):
    """
    Safely read uploaded CSV/XLSX into a DataFrame, without crashing the app.
    Returns (df, error_str). df may be None if failed.
    """
    if uploaded_file is None:
        return None, None

    name = (uploaded_file.name or "").lower()

    try:
        uploaded_file.seek(0)
    except Exception:
        pass

    try:
        if name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            return df, None
        elif name.endswith(".xlsx") or name.endswith(".xls"):
            try:
                df = pd.read_excel(uploaded_file)
                return df, None
            except ImportError as e:
                return None, (
                    "Reading Excel requires openpyxl. Add `openpyxl` to requirements.txt and redeploy.\n\n"
                    + str(e)
                )
        else:
            return None, f"Unsupported file type: {uploaded_file.name}"
    except Exception as e:
        return None, str(e)


def show_preview(df: pd.DataFrame, max_rows: int = 25):
    if df is None:
        return
    st.write("Columns detected:", list(df.columns))
    st.dataframe(df.head(max_rows), use_container_width=True)


def db_health_check():
    try:
        with engine.begin() as conn:
            one = conn.execute(text("SELECT 1")).scalar()
        st.success(f"DB OK (SELECT 1 → {one})")
    except Exception as e:
        st.error("DB connection failed")
        st.code(str(e))
        st.stop()


def safe_action_button(label: str, fn, *, key: str, args=(), kwargs=None):
    """
    Runs an action only when button pressed. Catches errors and reports them nicely.
    """
    if kwargs is None:
        kwargs = {}
    if st.button(label, key=key, type="primary"):
        try:
            res = fn(*args, **kwargs)
            st.success(res)
        except Exception as e:
            st.error("Action failed")
            st.code(str(e))


# -----------------------------
# DB Health
# -----------------------------
with st.expander("DB Health Check", expanded=True):
    db_health_check()

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(
    ["Load Qualifications/Units", "Load OSCA/ANZSCO + Mappings", "Build/Query Profiles"]
)

# =========================================================
# TAB 1: Qualifications + Units
# =========================================================
with tab1:
    st.subheader("Load Qualifications + Qualification→Units")
    st.caption(
        "Upload your Qualifications_*.csv OR a units-to-quals XLSX/CSV crosswalk (if it contains qualification_code + unit_code)."
    )

    q_file = st.file_uploader(
        "Upload Qualifications/Units file (CSV or XLSX)", type=["csv", "xlsx"], key="qual_file"
    )

    colA, colB, colC = st.columns(3)
    tp = colA.text_input("Training package (optional)", value="MSL")
    release = colB.text_input("Release (optional)", value="")
    source_file = colC.text_input("Source label (optional)", value="upload")

    df_preview, err = safe_read_dataframe(q_file)
    if q_file:
        if err:
            st.error("File preview failed")
            st.code(err)
        else:
            show_preview(df_preview, max_rows=25)

        st.divider()

        def _load_qual_units():
            q_file.seek(0)
            return load_qualifications_and_units(
                engine,
                q_file,
                q_file.name,
                training_package=tp or None,
                release=release or None,
                source_file=source_file or q_file.name,
            )

        safe_action_button(
            "Load Qualifications + Units into DB",
            _load_qual_units,
            key="btn_load_qual_units",
        )

# =========================================================
# TAB 2: Taxonomy + Mapping (auto-detect OSCA structure vs index)
# =========================================================
with tab2:
    st.subheader("Load OSCA / ANZSCO taxonomy and mappings")
    st.caption("OSCA Structure XLSX is auto-detected. OSCA Index CSV is auto-detected. Anything else uses the generic loader.")

    st.markdown("### Load taxonomy (auto-detect)")
    t_ver = st.text_input("Taxonomy version (optional)", value="", key="tax_ver")
    t_file = st.file_uploader("Upload taxonomy file (CSV or XLSX)", type=["csv", "xlsx"], key="tax_file")

    if t_file:
        name = (t_file.name or "").lower()

        # Show preview safely
        df_tax_preview, tax_err = safe_read_dataframe(t_file)
        if tax_err:
            st.warning("Preview not available (may be multi-sheet Excel).")
            st.code(tax_err)
        else:
            show_preview(df_tax_preview, max_rows=25)

        st.divider()

        if name.endswith(".xlsx") or name.endswith(".xls"):
            # OSCA structure detection
            try:
                t_file.seek(0)
                is_structure = is_osca_structure_xlsx(t_file, t_file.name)
            except Exception as e:
                is_structure = False
                st.warning("Could not auto-detect OSCA structure")
                st.code(str(e))

            if is_structure:
                st.success("Auto-detected: OSCA Structure (hierarchy) XLSX")

                def _load_osca_structure():
                    t_file.seek(0)
                    return load_osca_structure(
                        engine,
                        t_file,
                        t_file.name,
                        taxonomy_version=t_ver or None,
                        taxonomy_source="official",
                        sheet_name="Table 5",
                    )

                safe_action_button(
                    "Load OSCA Structure into DB",
                    _load_osca_structure,
                    key="btn_load_osca_structure",
                )
            else:
                st.info("Excel file not detected as OSCA Structure. Using generic taxonomy loader.")
                t_sys = st.selectbox("Taxonomy system", ["ANZSCO", "OSCA"], index=0, key="generic_tax_sys_xlsx")

                def _load_generic_tax_xlsx():
                    t_file.seek(0)
                    return load_occupation_taxonomy(
                        engine,
                        t_file,
                        t_file.name,
                        taxonomy_system=t_sys,
                        taxonomy_version=t_ver or None,
                        taxonomy_source="official",
                    )

                safe_action_button(
                    "Load taxonomy into DB (generic)",
                    _load_generic_tax_xlsx,
                    key="btn_load_generic_tax_xlsx",
                )

        else:
            # CSV may be OSCA index
            if df_tax_preview is not None and is_osca_index_titles(df_tax_preview):
                st.success("Auto-detected: OSCA Index of Principal/Alternative Titles (flat list)")

                def _load_osca_index():
                    t_file.seek(0)
                    return load_osca_index_titles(
                        engine,
                        t_file,
                        t_file.name,
                        taxonomy_version=t_ver or None,
                        taxonomy_source="official",
                    )

                safe_action_button(
                    "Load OSCA Index Titles into DB",
                    _load_osca_index,
                    key="btn_load_osca_index",
                )
            else:
                st.info("CSV not detected as OSCA Index. Using generic taxonomy loader.")
                t_sys = st.selectbox("Taxonomy system", ["ANZSCO", "OSCA"], index=0, key="generic_tax_sys_csv")

                def _load_generic_tax_csv():
                    t_file.seek(0)
                    return load_occupation_taxonomy(
                        engine,
                        t_file,
                        t_file.name,
                        taxonomy_system=t_sys,
                        taxonomy_version=t_ver or None,
                        taxonomy_source="official",
                    )

                safe_action_button(
                    "Load taxonomy into DB (generic)",
                    _load_generic_tax_csv,
                    key="btn_load_generic_tax_csv",
                )

    st.divider()
    st.markdown("### Load occupation → qualification map")
    oq_file = st.file_uploader(
        "Upload occupation→qualification map (CSV or XLSX)", type=["csv", "xlsx"], key="oq_file"
    )
    st.caption("Required columns: taxonomy_system, taxonomy_code, qualification_code. Optional: confidence_score, mapping_method, notes.")

    oq_df, oq_err = safe_read_dataframe(oq_file)
    if oq_file:
        if oq_err:
            st.error("File preview failed")
            st.code(oq_err)
        else:
            show_preview(oq_df, max_rows=25)

        def _load_occ_qual():
            oq_file.seek(0)
            return load_occupation_qualification_map(engine, oq_file, oq_file.name, default_mapping_method="manual")

        safe_action_button(
            "Load occupation→qualification map into DB",
            _load_occ_qual,
            key="btn_load_occ_qual",
        )

    st.divider()
    st.markdown("### Load ANZSCO ↔ OSCA crosswalk (optional)")
    cw_file = st.file_uploader("Upload crosswalk file (CSV or XLSX)", type=["csv", "xlsx"], key="cw_file")
    st.caption("Required columns: from_taxonomy_system, from_taxonomy_code, to_taxonomy_system, to_taxonomy_code.")

    cw_df, cw_err = safe_read_dataframe(cw_file)
    if cw_file:
        if cw_err:
            st.error("File preview failed")
            st.code(cw_err)
        else:
            show_preview(cw_df, max_rows=25)

        def _load_crosswalk():
            cw_file.seek(0)
            return load_occupation_crosswalk(engine, cw_file, cw_file.name, default_mapping_type="official")

        safe_action_button(
            "Load crosswalk into DB",
            _load_crosswalk,
            key="btn_load_crosswalk",
        )

# =========================================================
# TAB 3: Build / Query / Export
# =========================================================
with tab3:
    st.subheader("Build occupation skill profiles")
    st.caption("Profiles are built by inheritance: occupation → qualifications → units → skill_records (element skills).")

    run_id = st.text_input("Optional: constrain to run_id (leave blank to use all skill_records)", value="", key="run_filter")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Build one occupation")
        occ_id = st.text_input("occupation_id (e.g., OSCA:311411 or ANZSCO:311411)", value="", key="occ_one")
        replace = st.toggle("Replace existing profile rows for this occupation", value=True, key="replace_one")

        def _build_one():
            return build_occupation_skill_profile(
                engine,
                occ_id.strip(),
                run_id=run_id.strip() or None,
                replace=replace,
            )

        if st.button("Build profile for occupation", key="btn_build_one", type="primary"):
            if not occ_id.strip():
                st.error("Enter an occupation_id")
            else:
                try:
                    res = _build_one()
                    st.success(res)
                except Exception as e:
                    st.error("Profile build failed")
                    st.code(str(e))

    with col2:
        st.markdown("### Build all mapped occupations")
        replace_all = st.toggle("Replace existing profile rows for ALL occupations", value=False, key="replace_all")

        def _build_all():
            return build_all_occupation_profiles(
                engine,
                run_id=run_id.strip() or None,
                replace=replace_all,
            )

        safe_action_button(
            "Build profiles for all occupations",
            _build_all,
            key="btn_build_all",
        )

    st.divider()
    st.subheader("Query / Export occupation skill profiles")

    q_occ = st.text_input("Filter by occupation_id (optional)", value="", key="q_occ")
    q_limit = st.number_input("Limit", min_value=50, max_value=20000, value=2000, step=50, key="q_limit")

    if st.button("Run query", key="btn_query_profiles"):
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

        try:
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
        except Exception as e:
            st.error("Query failed")
            st.code(str(e))
