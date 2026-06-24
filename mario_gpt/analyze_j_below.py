import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mario_gpt.level import FULL_LEVEL_STR_WITH_PATHS

rows = [r for r in FULL_LEVEL_STR_WITH_PATHS.split("\n") if r.strip()]
H = len(rows)
W = len(rows[0])

solid_count = 0
air_count   = 0
last_row    = 0
details     = []  # (col, row, below_char)

for c in range(W):
    for r in range(H):
        if c >= len(rows[r]):
            continue
        if rows[r][c] != 'j':
            continue

        if r + 1 >= H:
            last_row += 1
            continue

        below = rows[r + 1][c] if c < len(rows[r + 1]) else '-'
        if below == '-':
            air_count += 1
            details.append((c, r, below))
        else:
            solid_count += 1

total = solid_count + air_count + last_row
print(f"j TOKEN 總數       : {total}")
print(f"  下方為實體 (非-)  : {solid_count}  ({solid_count/total*100:.1f}%)")
print(f"  下方為空氣 (-)    : {air_count}   ({air_count/total*100:.1f}%)")
print(f"  在最後一行 (無下方): {last_row}")
print()

# 統計下方空氣的 j 在第幾 row（高度分佈）
from collections import Counter
row_dist = Counter(r for c, r, b in details)
print("下方為 - 的 j，所在 row 分佈：")
for r in sorted(row_dist):
    print(f"  row {r:2d} : {row_dist[r]} 個")

print()
print("下方為 - 的 j，所有位置 (col, row)：")
for c, r, b in sorted(details):
    print(f"  col {c:4d}, row {r:2d}")
