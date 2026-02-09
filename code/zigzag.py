import os
import numpy as np


# =======================
# Configuration (Relative Paths)
# =======================

INPUT_DIR = r"..\results\dct_blocks_csv"
OUTPUT_DIR = r"..\results\zigzag_csv"


BLOCK_SIZES = [8, 16, 32, 64]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =======================
# Zigzag Utility
# =======================

def zigzag_indices(n):
    indices = []
    for s in range(2 * n - 1):
        if s % 2 == 0:
            for i in range(min(s, n - 1), max(-1, s - n), -1):
                j = s - i
                indices.append((i, j))
        else:
            for j in range(min(s, n - 1), max(-1, s - n), -1):
                i = s - j
                indices.append((i, j))
    return indices


# =======================
# Main Processing
# =======================

for filename in os.listdir(INPUT_DIR):
    if not filename.endswith(".csv"):
        continue

    # Extract n from filename: picname-n-block.csv
    parts = filename.split("-")
    n = int(parts[-2])

    if n not in BLOCK_SIZES:
        continue

    csv_path = os.path.join(INPUT_DIR, filename)
    dct_image = np.loadtxt(csv_path, delimiter=",")

    H, W = dct_image.shape
    blocks_per_row = W // n

    zz_idx = zigzag_indices(n)
    zigzag_rows = []

    # Traverse blocks row-major (top-left → right → down)
    for i in range(0, H, n):
        for j in range(0, W, n):
            block = dct_image[i:i+n, j:j+n]
            zigzag_vector = [block[x, y] for (x, y) in zz_idx]
            zigzag_rows.append(zigzag_vector)

    zigzag_matrix = np.array(zigzag_rows)

    # Save
    out_name = filename.replace("-block.csv", "-zigzag.csv")
    out_path = os.path.join(OUTPUT_DIR, out_name)
    np.savetxt(out_path, zigzag_matrix, delimiter=",")

    print(f"Saved zigzag CSV: {out_name}")

print("Zigzag conversion completed.")
