import os
import cv2
import numpy as np
from scipy.fftpack import dct

# =======================
# Configuration (Relative Paths)
# =======================

INPUT_DIR = r"..\data"
OUTPUT_DIR = r"..\results\dct_blocks_csv"

BLOCK_SIZES = [8, 16, 32, 64]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =======================
# DCT Functions
# =======================

def dct2(block):
    """
    Apply 2D DCT (Type-II) with orthonormal normalization
    """
    return dct(dct(block.T, norm='ortho').T, norm='ortho')


# =======================
# Main Processing
# =======================

for filename in os.listdir(INPUT_DIR):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
        continue

    img_path = os.path.join(INPUT_DIR, filename)

    # Load color image
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"Failed to load image: {filename}")
        continue

    # Convert to YCbCr (OpenCV uses YCrCb)
    img_ycbcr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)

    h, w, _ = img_ycbcr.shape

    for n in BLOCK_SIZES:

        assert h % n == 0 and w % n == 0, \
            f"Image size not divisible by n={n}"

        # Process each channel independently
        for ch_idx, ch_name in enumerate(["Y", "Cr", "Cb"]):

            channel = img_ycbcr[:, :, ch_idx]
            dct_image = np.zeros_like(channel, dtype=np.float32)

            for i in range(0, h, n):
                for j in range(0, w, n):
                    block = channel[i:i+n, j:j+n]
                    dct_block = dct2(block)
                    dct_image[i:i+n, j:j+n] = dct_block

            base_name = os.path.splitext(filename)[0]
            out_name = f"{base_name}-{n}-{ch_name}-block.csv"
            out_path = os.path.join(OUTPUT_DIR, out_name)

            np.savetxt(out_path, dct_image, delimiter=",")

            print(f"Saved: {out_name}")

print("Block-wise DCT processing completed for all n and channels.")
