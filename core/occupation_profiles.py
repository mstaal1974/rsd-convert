import json
from datetime import datetime, timezone
from typing import Optional, Dict, Any

from sqlalchemy import text
from sqlalchemy.engine import Engine


DEFAULT_WEIGHTS = {
    "core": 1.0,
    "elective": 0.6,
    "specialisation": 0.8,
    "unknown": 0.7,
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_occupation_skill_profile(
    engine: Engine,
    occupation_id: str,
    *,
    # optional: constrain to a particular run_id (otherwise uses “any run”)
    run_id: Optional[str] = None,
    # weighting override
    weights: Optional[Dict[str, float]] = None,
    # if True, wipe existing profile rows for that occupation first
    replace: bool = True,
) -> Dict[str, Any]:
    """
    Materialise Occupation → Skills by inheriting:
      occupation -> occupation_qualifications -> qualification_units -> skill_records

    Returns summary dict.
    """
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)

    with engine.begin() as conn:
        # Validate occupation exists
        occ = conn.execute(
            text("SELECT occupation_id, occupation_name FROM occupations WHERE occupation_id=:oid"),
            {"oid": occupation_id},
        ).mappings().first()
        if not occ:
            raise ValueError(f"Unknown occupation_id: {occupation_id}")

        if replace:
            conn.execute(
                text("DELETE FROM occupation_skill_profiles WHERE occupation_id=:oid"),
                {"oid": occupation_id},
            )

        # Strategy for selecting skill_records:
        # - If run_id provided: use that run_id only.
        # - Else: use all skill_records (most teams will generate one run per package anyway)
        # You can later upgrade this to “latest run per unit_code”.
        run_filter_sql = ""
        params = {"oid": occupation_id}
        if run_id:
            run_filter_sql = "AND s.run_id = :run_id"
            params["run_id"] = run_id

        # Build derived rows
        rows = conn.execute(
            text(f"""
            WITH oq AS (
              SELECT occupation_id, qualification_code, mapping_method, confidence_score
              FROM occupation_qualifications
              WHERE occupation_id = :oid
            ),
            qu AS (
              SELECT
                oq.occupation_id,
                oq.qualification_code,
                q.qualification_title,
                q.training_package,
                q.release,
                qu.unit_code,
                COALESCE(qu.unit_type, 'core') AS unit_type
              FROM oq
              JOIN qualifications q
                ON q.qualification_code = oq.qualification_code
              JOIN qualification_units qu
                ON qu.qualification_code = oq.qualification_code
            ),
            skills AS (
              SELECT
                qu.occupation_id,
                qu.qualification_code,
                qu.unit_code,
                qu.unit_type,
                s.record_id
              FROM qu
              JOIN skill_records s
                ON s.unit_code = qu.unit_code
              WHERE 1=1
              {run_filter_sql}
            )
            SELECT * FROM skills
            """),
            params,
        ).mappings().all()

        inserted = 0
        if not rows:
            return {
                "occupation_id": occupation_id,
                "occupation_name": occ["occupation_name"],
                "status": "no_rows",
                "inserted": 0,
                "run_id": run_id,
                "generated_at_utc": utc_now_iso(),
            }

        # Upsert profile rows
        upsert_sql = text("""
        INSERT INTO occupation_skill_profiles (
          occupation_id, record_id, unit_code, qualification_code,
          unit_type, weight, confidence_score, provenance_json,
          created_at_utc, updated_at_utc
        )
        VALUES (
          :occupation_id, :record_id, :unit_code, :qualification_code,
          :unit_type, :weight, :confidence_score, CAST(:provenance_json AS JSONB),
          now(), now()
        )
        ON CONFLICT (occupation_id, record_id, qualification_code) DO UPDATE SET
          unit_code = EXCLUDED.unit_code,
          unit_type = EXCLUDED.unit_type,
          weight = EXCLUDED.weight,
          confidence_score = EXCLUDED.confidence_score,
          provenance_json = EXCLUDED.provenance_json,
          updated_at_utc = now()
        """)

        payload = []
        for r in rows:
            unit_type = (r["unit_type"] or "unknown").strip().lower()
            weight = float(w.get(unit_type, w["unknown"]))

            prov = {
                "method": "inheritance",
                "generated_at_utc": utc_now_iso(),
                "run_id": run_id,
                "weight_rule": {"unit_type": unit_type, "weight": weight},
            }

            payload.append(
                {
                    "occupation_id": occupation_id,
                    "record_id": r["record_id"],
                    "unit_code": r["unit_code"],
                    "qualification_code": r["qualification_code"],
                    "unit_type": unit_type,
                    "weight": weight,
                    "confidence_score": None,  # reserved for future
                    "provenance_json": json.dumps(prov, ensure_ascii=False),
                }
            )

        conn.execute(upsert_sql, payload)
        inserted = len(payload)

        return {
            "occupation_id": occupation_id,
            "occupation_name": occ["occupation_name"],
            "status": "ok",
            "inserted": inserted,
            "run_id": run_id,
            "generated_at_utc": utc_now_iso(),
        }


def build_all_occupation_profiles(
    engine: Engine,
    *,
    run_id: Optional[str] = None,
    replace: bool = True,
) -> Dict[str, Any]:
    """
    Build profiles for every occupation that has at least one qualification mapping.
    """
    with engine.begin() as conn:
        occs = conn.execute(
            text("""
            SELECT DISTINCT o.occupation_id
            FROM occupations o
            JOIN occupation_qualifications oq
              ON oq.occupation_id = o.occupation_id
            ORDER BY o.occupation_id
            """)
        ).scalars().all()

    results = []
    for oid in occs:
        results.append(build_occupation_skill_profile(engine, oid, run_id=run_id, replace=replace))

    return {
        "status": "ok",
        "count": len(results),
        "results": results,
    }
