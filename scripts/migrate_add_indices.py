"""
RoboAlgo — One-time migration: add missing DB indices.

Run once after deploying the audit fix:
    python scripts/migrate_add_indices.py

Uses CREATE INDEX CONCURRENTLY IF NOT EXISTS — safe on a live DB.
Falls back to non-concurrent CREATE INDEX IF NOT EXISTS in the rare
case the DB doesn't support CONCURRENTLY (e.g. in a transaction block).
"""
from __future__ import annotations

import logging
from database.connection import get_engine

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

INDICES = [
    # PatternDetection — new indices for confluence engine queries
    ("ix_pattern_status",       "pattern_detections", "status"),
    ("ix_pattern_confidence",   "pattern_detections", "confidence"),
    # Composite: WHERE instrument_id=? ORDER BY date DESC, confidence DESC
    ("ix_pattern_inst_date_conf",
     "pattern_detections",
     "instrument_id, date DESC, confidence DESC"),

    # ConfluenceScore — composite indices for dashboard hot paths
    ("ix_confluence_date_score",
     "confluence_scores",
     "date DESC, confluence_score DESC"),
    ("ix_confluence_tier_score",
     "confluence_scores",
     "signal_tier, confluence_score DESC"),
]


def run():
    engine = get_engine()
    with engine.connect() as conn:
        for name, table, cols in INDICES:
            try:
                sql = (
                    f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {name} "
                    f"ON {table} ({cols})"
                )
                conn.execute(__import__("sqlalchemy").text(sql))
                conn.commit()
                log.info(f"  ✓  {name}  ON  {table} ({cols})")
            except Exception as exc:
                # CONCURRENTLY not allowed inside a transaction — retry without
                try:
                    conn.rollback()
                    sql2 = (
                        f"CREATE INDEX IF NOT EXISTS {name} "
                        f"ON {table} ({cols})"
                    )
                    conn.execute(__import__("sqlalchemy").text(sql2))
                    conn.commit()
                    log.info(f"  ✓  {name}  (non-concurrent fallback)")
                except Exception as exc2:
                    log.warning(f"  ✗  {name}  →  {exc2}")

    log.info("Migration complete.")


if __name__ == "__main__":
    run()
