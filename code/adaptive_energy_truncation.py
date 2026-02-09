import os
import numpy as np

# =======================
# Configuration
# =======================

INPUT_DIR  = r"..\results\zigzag_csv"
OUTPUT_DIR = r"..\results\truncated_csv"

T_VALUES = np.arange(0.75, 1.01, 0.03)  # 75% → 100%

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =======================
# Core Function
# =======================

def energy_based_truncation(zigzag_vector, T):
    """
    Perform adaptive energy-based truncation on one zigzag vector
    Returns:
        k : number of coefficients kept
        truncated_vector : zeroed vector
    """
    energy = np.abs(zigzag_vector) ** 2
    total_energy = np.sum(energy)

    cumulative_energy = np.cumsum(energy)
    threshold = T * total_energy

    # Minimum k satisfying energy condition
    k = np.searchsorted(cumulative_energy, threshold) + 1

    truncated = np.zeros_like(zigzag_vector)
    truncated[:k] = zigzag_vector[:k]

    return k, truncated


# =======================
# Main Processing
# =======================

for filename in os.listdir(INPUT_DIR):
    if not filename.endswith(".csv"):
        continue

    input_path = os.path.join(INPUT_DIR, filename)
    zigzag_data = np.loadtxt(input_path, delimiter=",")

    base_name = filename.replace("-zigzag.csv", "")

    for T in T_VALUES:
        rows = []
        k_values = []

        for idx, block in enumerate(zigzag_data):
            k, truncated_block = energy_based_truncation(block, T)
            k_values.append(k)
            print(f"Block {idx:5d} | k = {k}")

            row = np.concatenate(([k], truncated_block))
            rows.append(row)


        rows = np.array(rows)
        k_values = np.array(k_values)

        T_percent = int(T * 100)
        out_name = f"{base_name}-T{T_percent}-zigzag-truncated.csv"
        out_path = os.path.join(OUTPUT_DIR, out_name)

        np.savetxt(out_path, rows, delimiter=",")

        print(
            f"Saved: {out_name} | "
            f"k(min/mean/max) = "
            f"{k_values.min()} / "
            f"{k_values.mean():.2f} / "
            f"{k_values.max()}"
        )


print("Adaptive energy-based truncation completed.")
