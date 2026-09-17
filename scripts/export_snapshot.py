import json
import os
import sqlite3
import sys

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def export_snapshot(db_path: str = "data/ghostjobs.db", out_path: str = "data/benchmark_snapshot.json"):
    if not os.path.exists(db_path):
        print(f"Database {db_path} does not exist.")
        return False

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    tables = ["job_postings", "ghost_scores", "company_reviews", "company_sentiments", "page_analytics"]
    snapshot = {}

    for table in tables:
        try:
            cur.execute(f"SELECT * FROM {table}")
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
            snapshot[table] = rows
            print(f"Exported {len(rows)} records from '{table}'.")
        except Exception as e:
            print(f"Could not export table {table}: {e}")
            snapshot[table] = []

    conn.close()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, default=str)

    print(f"Successfully wrote snapshot to '{out_path}' ({os.path.getsize(out_path)} bytes).")
    return True

if __name__ == "__main__":
    export_snapshot()
