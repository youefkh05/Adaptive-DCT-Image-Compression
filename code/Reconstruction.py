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

H, W = 512, 768   # Known image size

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


def reconstruct_channel(csv_path, n):
    """
    Reconstruct a single Y / Cr / Cb channel from truncated zigzag CSV
    """
    data = np.loadtxt(csv_path, delimiter=",")
    coeffs = data[:, 1:]  # drop k

    channel = np.zeros((H, W), dtype=np.float32)

    block_idx = 0
    for i in range(0, H, n):
        for j in range(0, W, n):
            zigzag_vec = coeffs[block_idx]
            dct_block = inverse_zigzag(zigzag_vec, n)
            spatial_block = idct2(dct_block)
            channel[i:i+n, j:j+n] = spatial_block
            block_idx += 1

    return channel


# =======================
# Main Reconstruction
# =======================

files = [f for f in os.listdir(INPUT_DIR) if f.endswith("-zigzag-truncated.csv")]

# Group files by (pic, n, T)
groups = {}

for f in files:
    # Format: pic-n-channel-T99-zigzag-truncated.csv
    parts = f.replace(".csv", "").split("-")
    pic_id  = parts[0]
    n       = int(parts[1])
    channel = parts[2]
    T       = parts[3]  # T99

    key = (pic_id, n, T)
    groups.setdefault(key, {})[channel] = f


for (pic_id, n, T), channel_files in groups.items():

    # Ensure all 3 channels exist
    if not all(ch in channel_files for ch in ["Y", "Cr", "Cb"]):
        print(f"Skipping incomplete set: {pic_id}, n={n}, {T}")
        continue

    print(f"Reconstructing RGB image: {pic_id}, n={n}, {T}")

    Y  = reconstruct_channel(os.path.join(INPUT_DIR, channel_files["Y"]),  n)
    Cr = reconstruct_channel(os.path.join(INPUT_DIR, channel_files["Cr"]), n)
    Cb = reconstruct_channel(os.path.join(INPUT_DIR, channel_files["Cb"]), n)

    # Stack into YCrCb image
    ycrcb = np.stack([Y, Cr, Cb], axis=2)

    # Clip and convert to uint8
    ycrcb = np.clip(ycrcb, 0, 255).astype(np.uint8)

    # Convert back to BGR (OpenCV RGB)
    bgr = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    out_name = f"{pic_id}-{n}-{T}-reconstructed.png"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    cv2.imwrite(out_path, bgr)

    print(f"Saved RGB image: {out_name}")

print("RGB reconstruction completed.")
