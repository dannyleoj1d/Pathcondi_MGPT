import os

INPUT_PATH  = os.path.join(os.path.dirname(__file__), "..", "level_9_A.txt")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "level_9_A_P.txt")

PATH_TOKENS = frozenset({'M', 'J', 'j', 'A', 'C'})

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    lines = f.read().splitlines()

rows = [ln for ln in lines if ln]
H = len(rows)
W = max(len(r) for r in rows)

# Build mutable grid
grid = []
for r in range(H):
    row = list(rows[r].ljust(W, '-'))
    grid.append(row)

# For each column: find topmost path token row,
# then convert '-' → 'P' for every row below it.
for c in range(W):
    top_path_row = next(
        (r for r in range(H) if grid[r][c] in PATH_TOKENS), None
    )
    if top_path_row is None:
        continue  # no path token in this column → leave unchanged
    for r in range(top_path_row + 1, H):
        if grid[r][c] == '-':
            grid[r][c] = 'P'

result = '\n'.join(''.join(grid[r]) for r in range(H)) + '\n'

with open(OUTPUT_PATH, "w", encoding="utf-8", newline="\n") as f:
    f.write(result)

print(f"Done: {OUTPUT_PATH}")
