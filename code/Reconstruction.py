import os
import numpy as np
import cv2
from scipy.fftpack import idct

# =======================
# Configuration
# =======================

INPUT_DIR  = r"..\results\truncated_csv"
OUTPUT_DIR = r"..\results\reconstructed_images"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =======================
# Utilities
# =======================

def inverse_zigzag(vector, n):
    block = np.zeros((n, n))
    idx = 0
    for s in range(2 * n - 1):
        if s % 2 == 0:
            for i in range(min(s, n - 1), max(-1, s - n), -1):
                j = s - i
                block[i, j] = vector[idx]
                idx += 1
        else:
            for j in range(min(s, n - 1), max(-1, s - n), -1):
                i = s - j
                block[i, j] = vector[idx]
                idx += 1
    return block


def idct2(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')


# =======================
# Main Reconstruction
# =======================

for filename in os.listdir(INPUT_DIR):
    if not filename.endswith(".csv"):
        continue

    # Example: 1-8-T75-zigzag-truncated.csv
    parts = filename.split("-")
    pic_name = parts[0]
    n = int(parts[1])

    csv_path = os.path.join(INPUT_DIR, filename)
    data = np.loadtxt(csv_path, delimiter=",")

    num_blocks = data.shape[0]
    coeffs = data[:, 1:]  # drop k column

    # Determine image size
    # You already know images are 512 x 768
    H, W = 512, 768
    reconstructed = np.zeros((H, W), dtype=np.float32)

    blocks_per_row = W // n

    block_idx = 0
    for i in range(0, H, n):
        for j in range(0, W, n):
            zigzag_vec = coeffs[block_idx]
            dct_block = inverse_zigzag(zigzag_vec, n)
            spatial_block = idct2(dct_block)
            reconstructed[i:i+n, j:j+n] = spatial_block
            block_idx += 1

    # Clip to valid image range
    reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)

    out_name = filename.replace("-zigzag-truncated.csv", "-reconstructed.png")
    out_path = os.path.join(OUTPUT_DIR, out_name)
    cv2.imwrite(out_path, reconstructed)

    print(f"Reconstructed image saved: {out_name}")

print("Reconstruction completed.")
