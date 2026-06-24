import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mario_gpt.level import FULL_LEVEL_STR_WITH_PATHS

PATH_TOKENS = frozenset({'M', 'J', 'j', 'A', 'C'})

rows = [r for r in FULL_LEVEL_STR_WITH_PATHS.split("\n") if r.strip()]
H = len(rows)
W = len(rows[0])

empty_cols = []
for c in range(W):
    has_path = any(c < len(rows[r]) and rows[r][c] in PATH_TOKENS for r in range(H))
    if not has_path:
        empty_cols.append(c)

print(f"Level size: {W} cols x {H} rows")
print(f"Columns with NO path token: {len(empty_cols)}")
print(f"Column indices: {empty_cols}")

# 顯示連續區間
if empty_cols:
    ranges = []
    start = empty_cols[0]
    prev  = empty_cols[0]
    for c in empty_cols[1:]:
        if c == prev + 1:
            prev = c
        else:
            ranges.append((start, prev))
            start = prev = c
    ranges.append((start, prev))
    print("Ranges:")
    for s, e in ranges:
        print(f"  col {s} ~ {e}  (length {e - s + 1})")
