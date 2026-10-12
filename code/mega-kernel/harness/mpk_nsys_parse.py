"""Print the ordered CUDA runtime API trace from an nsys sqlite export.

Usage: python3 mpk_nsys_parse.py <file.sqlite> [--tail N]
Compares host-side launch sequencing between the HANG (prewarm=none) and PASS
(prewarm=cast_1elem) runs. Only the CUDA runtime API (cuda*/cu*) rows are the
interesting part of the launch/handshake path.
"""
import sqlite3
import sys


def main():
    path = sys.argv[1]
    tail = None
    if "--tail" in sys.argv:
        tail = int(sys.argv[sys.argv.index("--tail") + 1])
    c = sqlite3.connect(path)
    have = {n for (n,) in c.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    if "CUPTI_ACTIVITY_KIND_RUNTIME" not in have:
        print(f"# {path}: NO runtime API table (CUPTI buffer not flushed; "
              f"tables={len(have)})")
        return
    rows = []
    q = ("SELECT start, nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME "
         "ORDER BY start ASC")
    for start, name_id in c.execute(q):
        rows.append((start, name_id))
    ids = {i: n for i, n in c.execute("SELECT id, value FROM StringIds")}
    t0 = rows[0][0] if rows else 0
    if tail:
        rows = rows[-tail:]
    print(f"# {path}: {len(rows)} runtime API calls (relative ns)")
    for start, name_id in rows:
        print(f"  +{(start - t0) / 1e6:12.3f} ms  {ids.get(name_id, name_id)}")


if __name__ == "__main__":
    main()
