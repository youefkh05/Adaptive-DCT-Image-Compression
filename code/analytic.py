import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import csv

# =======================
# Configuration
# =======================

DATA_DIR       = r"..\data"
TRUNCATED_DIR  = r"..\results\truncated_csv"
RECON_DIR      = r"..\results\reconstructed_images"
OUTPUT_DIR     = r"..\results\analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)

H, W = 512, 768

RESULTS_CSV = os.path.join(OUTPUT_DIR, "metrics.csv")

# =======================
# Metrics
# =======================

def compute_cr_dct(k_values):
    return (H * W) / np.sum(k_values)

def compute_psnr(original, reconstructed):
    mse = np.mean((original.astype(np.float32) -
                    reconstructed.astype(np.float32)) ** 2)
    if mse == 0:
        return np.inf
    return 10 * np.log10((255 ** 2) / mse)

def file_size_kb(path):
    return os.path.getsize(path) / 1024.0

def compute_cr_png(orig_path, recon_path):
    return file_size_kb(orig_path) / file_size_kb(recon_path)

# =======================
# Heatmap
# =======================

def generate_k_heatmap(k_values, n):
    heatmap = np.zeros((H, W))
    idx = 0
    for i in range(0, H, n):
        for j in range(0, W, n):
            heatmap[i:i+n, j:j+n] = k_values[idx]
            idx += 1
    return heatmap

# =======================
# Prepare CSV
# =======================

with open(RESULTS_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Image", "BlockSize", "T",
        "CR_DCT", "CR_PNG", "PSNR_Y"
    ])

# =======================
# Main Analysis Loop
# =======================

for file in os.listdir(TRUNCATED_DIR):

    if not file.endswith("-zigzag-truncated.csv"):
        continue

    # Format:
    # pic-n-Y-T75-zigzag-truncated.csv
    parts = file.replace(".csv", "").split("-")

    pic_id = parts[0]
    n      = int(parts[1])
    channel= parts[2]
    T      = parts[3]   # T75, T78, ...

    # ---- Only analyze Y channel
    if channel != "Y":
        continue

    print(f"\nAnalyzing Image {pic_id}, n={n}, {T}")

    trunc_path = os.path.join(TRUNCATED_DIR, file)
    data = np.loadtxt(trunc_path, delimiter=",")

    k_values = data[:, 0]

    # ---- CR (DCT)
    cr_dct = compute_cr_dct(k_values)

    # ---- Original Y
    orig_bgr = cv2.imread(os.path.join(DATA_DIR, f"{pic_id}.png"))
    orig_y = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0]

    # ---- Reconstructed Y
    recon_path = os.path.join(
        RECON_DIR, f"{pic_id}-{n}-{T}-reconstructed.png"
    )
    recon_bgr = cv2.imread(recon_path)
    recon_y = cv2.cvtColor(recon_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0]

    # ---- PSNR
    psnr_y = compute_psnr(orig_y, recon_y)

    # ---- CR (PNG)
    cr_png = compute_cr_png(
        os.path.join(DATA_DIR, f"{pic_id}.png"),
        recon_path
    )

    print(f"CR_DCT={cr_dct:.2f}, CR_PNG={cr_png:.2f}, PSNR_Y={psnr_y:.2f}")

    # ---- Save metrics
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            pic_id, n, T,
            f"{cr_dct:.3f}",
            f"{cr_png:.3f}",
            f"{psnr_y:.3f}"
        ])

    # ---- Heatmap
    heatmap = generate_k_heatmap(k_values, n)

    plt.figure(figsize=(6, 4))
    plt.imshow(heatmap, cmap="hot")
    plt.colorbar(label="k (coefficients kept)")
    plt.title(f"Heatmap (Y): Image {pic_id}, n={n}, {T}")
    plt.axis("off")

    heatmap_path = os.path.join(
        OUTPUT_DIR, f"{pic_id}-{n}-{T}-heatmap.png"
    )
    plt.savefig(heatmap_path, dpi=300)
    plt.close()

    print(f"Saved heatmap: {heatmap_path}")

print("\nAnalysis completed successfully.")
