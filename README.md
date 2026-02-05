# Adaptive DCT-Based Image Compression  
**DSP Applications Project**

## 📌 Project Overview
This project implements an adaptive image compression framework based on the **2D Discrete Cosine Transform (2D-DCT)**.  
Instead of retaining a fixed number of transform coefficients, the system applies an **energy-targeted strategy**, preserving the minimum number of coefficients required to reach a specified percentage of the total block energy.

The project focuses on analyzing:
- Energy compaction properties of the DCT
- Spatial variation of information density in images
- The rate–distortion tradeoff between compression ratio and reconstruction quality

---

## 🎯 Objectives
- Demonstrate **energy compaction** in the DCT domain
- Visualize **spatial entropy** using coefficient heatmaps
- Analyze the relationship between:
  - Compression Ratio (CR)
  - Peak Signal-to-Noise Ratio (PSNR)
- Study the effect of **block size** and **image content** on compression efficiency

---

## ⚙️ Implementation Pipeline

### 1. Image Partitioning
- Images are converted to grayscale
- Partitioned into non-overlapping blocks of size `n × n`
- Block sizes tested: `n ∈ {8, 16, 32, 64}`
- Images are cropped or zero-padded if needed

### 2. Frequency Domain Transformation
- Apply **2D-DCT** to each block
- Coefficients are reordered using **zigzag scanning**
  - From low-frequency (DC) to high-frequency (AC)

### 3. Adaptive Energy-Based Truncation
For each block:
- Compute total energy:
  
  \[
  E_{total} = \sum |C_{i,j}|^2
  \]

- Retain the minimum number of coefficients `k` such that:

  \[
  \sum_{i=1}^{k} |C_i|^2 \geq T\% \cdot E_{total}
  \]

- Truncation threshold `T` varies from **75% to 100%**
- Remaining coefficients are set to zero
- The value of `k` is stored for heatmap visualization

### 4. Reconstruction
- Apply inverse zigzag scanning
- Perform **2D-Inverse DCT (IDCT)**
- Reassemble image blocks to form the reconstructed image

---

## 📊 Performance Metrics

### Compression Ratio (CR)
\[
CR = \frac{\text{Total number of pixels}}{\sum k \text{ (over all blocks)}}
\]

### Peak Signal-to-Noise Ratio (PSNR)
\[
PSNR = 10 \log_{10}\left(\frac{255^2}{MSE}\right)
\]

Where `MSE` is the mean squared error between the original and reconstructed images.

---

## 📈 Visual Results
- Original vs reconstructed image comparisons for high and low energy thresholds
- Coefficient heatmaps showing spatial variation in retained coefficients
- PSNR vs Compression Ratio plots for:
  - Different image contents
  - Different block sizes

---

## 🧠 Key Observations
- Smooth regions require fewer coefficients to preserve energy
- Edge and textured regions require more coefficients
- Larger block sizes improve frequency resolution but introduce **blocking artifacts**
- PSNR drops sharply as high-frequency coefficients are discarded

---

## 🛠️ Tools & Libraries
- Python 3
- NumPy
- SciPy
- OpenCV
- Matplotlib

## 📁 Repository Structure
- `data/`       : Input images and preprocessing outputs  
- `code/`       : Core DSP implementation  
- `results/`    : Reconstructed images, heatmaps, and plots  
- `report/`     : Final report and figures  


---

## 👥 Team
DSP Applications Project  
Faculty of Engineering

