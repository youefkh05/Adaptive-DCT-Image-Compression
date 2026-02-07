import os
import cv2
import numpy as np

# =======================
# Configuration
# =======================
INPUT_DIR = r"D:\project\DSP\Adaptive-DCT-Image-Compression\data"
OUTPUT_DIR = r"D:\project\DSP\Adaptive-DCT-Image-Compression\data\pre"

BLOCK_SIZES = [8, 16, 32, 64]

# Create output directory if it does not exist
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =======================
# Helper Functions
# =======================

def zero_pad_image(img, n):
    """
    Zero-pad image so that its dimensions are divisible by n
    """
    h, w = img.shape
    pad_h = (n - (h % n)) % n
    pad_w = (n - (w % n)) % n

    padded_img = np.pad(
        img,
        ((0, pad_h), (0, pad_w)),
        mode='constant',
        constant_values=0
    )

    return padded_img


def crop_image(img, n):
    """
    Crop image so that its dimensions are divisible by n
    """
    h, w = img.shape
    cropped_h = h - (h % n)
    cropped_w = w - (w % n)

    return img[:cropped_h, :cropped_w]


# =======================
# Main Preprocessing Loop
# =======================

for filename in os.listdir(INPUT_DIR):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
        continue

    img_path = os.path.join(INPUT_DIR, filename)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Failed to load image: {filename}")
        continue

    h, w = img.shape
    print(f"\nImage: {filename} | Original size: {h} x {w}")

    for n in BLOCK_SIZES:
        divisible = (h % n == 0) and (w % n == 0)

        # Case 1: Image already divisible
        if divisible:
            out_name = f"{n}-original-{filename}"
            out_path = os.path.join(OUTPUT_DIR, out_name)
            cv2.imwrite(out_path, img)

        # Case 2: Zero padding
        padded_img = zero_pad_image(img, n)
        ph, pw = padded_img.shape
        out_name_pad = f"{n}-zero-{filename}"
        out_path_pad = os.path.join(OUTPUT_DIR, out_name_pad)
        cv2.imwrite(out_path_pad, padded_img)

        # Case 3: Cropping
        cropped_img = crop_image(img, n)
        ch, cw = cropped_img.shape
        out_name_crop = f"{n}-crop-{filename}"
        out_path_crop = os.path.join(OUTPUT_DIR, out_name_crop)
        cv2.imwrite(out_path_crop, cropped_img)

        # ---- SIZE REPORTING ----
        print(f"  n = {n:2d} | Zero-padded: {ph} x {pw} | Cropped: {ch} x {cw}")

    print(f"Processed: {filename}")

print("\nPreprocessing completed successfully.")
