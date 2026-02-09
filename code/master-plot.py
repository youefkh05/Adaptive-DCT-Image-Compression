import os
import pandas as pd
import matplotlib.pyplot as plt

# =======================
# Configuration
# =======================

METRICS_CSV = r"..\results\analysis\metrics.csv"
OUTPUT_DIR  = r"..\results\analysis\master_plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =======================
# Load Data
# =======================

df = pd.read_csv(METRICS_CSV)

# Convert T from "T75" → 75
df["T_val"] = df["T"].str.replace("T", "").astype(int)

block_sizes = sorted(df["BlockSize"].unique())
images = sorted(df["Image"].unique())

# =======================
# Plot A: Content Sensitivity (Fixed n)
# =======================

for n in block_sizes:

    plt.figure(figsize=(7, 5))

    for img in images:
        subset = df[
            (df["BlockSize"] == n) &
            (df["Image"] == img)
        ].sort_values("T_val")

        plt.plot(
            subset["CR_DCT"],
            subset["PSNR_Y"],
            marker="o",
            label=f"Image {img}"
        )

    plt.xlabel("Compression Ratio (CR_DCT)")
    plt.ylabel("PSNR (dB)")
    plt.title(f"Content Sensitivity (Fixed n = {n})")
    plt.legend()
    plt.grid(True)

    out_path = os.path.join(
        OUTPUT_DIR, f"content_sensitivity_n{n}.png"
    )
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved: {out_path}")

# =======================
# Plot B: Block Size Sensitivity (Fixed Image)
# =======================

for img in images:

    plt.figure(figsize=(7, 5))

    for n in block_sizes:
        subset = df[
            (df["Image"] == img) &
            (df["BlockSize"] == n)
        ].sort_values("T_val")

        plt.plot(
            subset["CR_DCT"],
            subset["PSNR_Y"],
            marker="o",
            label=f"n = {n}"
        )

    plt.xlabel("Compression Ratio (CR_DCT)")
    plt.ylabel("PSNR (dB)")
    plt.title(f"Block Size Sensitivity (Image {img})")
    plt.legend()
    plt.grid(True)

    out_path = os.path.join(
        OUTPUT_DIR, f"blocksize_sensitivity_img{img}.png"
    )
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved: {out_path}")

print("\nAll master plots generated correctly.")
