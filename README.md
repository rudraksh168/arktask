# 🎥 Rod / Thin Object Medial Axis Tracker

A real-time computer vision pipeline that extracts the **medial axis** (centreline) of thin elongated objects from video — using background subtraction, morphological skeletonisation, and a custom Hough transform with edge-pairing logic.

---

## Pipeline Overview

```
Raw Video
   │
   ▼
[Stage 1] Colour Median Background Subtraction  →  diff video
   │
   ▼
[Stage 2] Morphological Cleaning                →  morph video
   │
   ▼
[Stage 3] Medial Axis / Skeletonisation         →  skeleton video
   │
   ▼
[Stage 4] Custom Hough Transform + Edge Pairing →  annotated video
```

---

## Stage 1 — Colour Median Background Model

### Building the Background

Instead of a single reference frame (fragile) or a running average (slow to converge), we sample **30 evenly-spaced frames** across the entire video and compute a **per-pixel, per-channel median**:

$$
B(x, y, c) = \text{median}\Bigl\{ f_1(x,y,c),\; f_2(x,y,c),\; \ldots,\; f_{30}(x,y,c) \Bigr\}
$$

where $c \in \{B, G, R\}$ and $f_i$ is the $i$-th sampled frame.

**Why median?** The median is robust to outliers. If the object appears in any given sample frame, it affects at most 1 in 30 values, so the median still converges to the true background pixel.

### Foreground Detection

For each new frame $F$, compute the absolute colour difference:

$$
D(x,y,c) = \bigl|F(x,y,c) - B(x,y,c)\bigr|
$$

This 3-channel difference is collapsed to a scalar by converting BGR → Grayscale (a weighted luminance sum):

$$
D_{\text{gray}}(x,y) = 0.114\, D_B + 0.587\, D_G + 0.299\, D_R
$$

Then apply a hard threshold $\tau = 30$:

$$
M(x,y) = \begin{cases} 255 & \text{if } D_{\text{gray}}(x,y) > \tau \\ 0 & \text{otherwise} \end{cases}
$$

---

## Stage 2 — Morphological Cleaning

Raw binary masks contain salt-and-pepper noise and small holes. We apply three morphological operations in sequence using a $5 \times 5$ elliptical structuring element $K$:

### 1. Opening (noise removal)

$$
M_{\text{open}} = (M \ominus K) \oplus K
$$

Opening = erosion followed by dilation. It **removes small isolated foreground blobs** that are smaller than $K$, while preserving the main object shape.

### 2. Closing (hole filling)

$$
M_{\text{close}} = (M_{\text{open}} \oplus K) \ominus K
$$

Closing = dilation followed by erosion. It **fills small dark holes** inside the foreground region.

### 3. Dilation (edge smoothing)

$$
M_{\text{clean}} = M_{\text{close}} \oplus K
$$

A final dilation slightly fattens the mask, which gives the skeletoniser a cleaner, more connected input — reducing fragmented skeleton branches.

---

## Stage 3 — Medial Axis via Morphological Thinning

OpenCV has no built-in medial axis function, so we implement **iterative morphological thinning** (Zhang–Suen family):

### Algorithm

Using a $3 \times 3$ cross-shaped structuring element $E$:

$$
T_i = I_i - \text{dilate}(\text{erode}(I_i, E),\; E) = I_i - (I_i \circ E)
$$

$$
S \mathrel{+}= T_i, \qquad I_{i+1} = \text{erode}(I_i, E)
$$

Repeat until $I_i$ is completely eroded to zero.

### What this computes

At each iteration:
- $\text{erode}(I_i, E)$ **peels away** one layer of boundary pixels
- $\text{dilate}(\text{erode}(I_i, E), E)$ approximately **restores** what was there
- Their **difference** $T_i$ captures exactly the pixels that were on the outer boundary

Accumulating all $T_i$ into $S$ gives the **skeleton** — the set of all points that are equidistant from at least two boundary edges, i.e., the true medial axis.

For a thin rod of half-width $r$, the skeleton converges to the centreline after approximately $r$ iterations.

---

## Stage 4 — Custom Hough Line Transform with Edge Pairing

### Standard Hough Transform — Maths

Every edge pixel $(x_i, y_i)$ in the skeleton votes for all lines passing through it. A line in **normal (Hesse) form** is:

$$
\rho = x\cos\theta + y\sin\theta, \quad \rho \in [-d, d],\; \theta \in [0, \pi)
$$

where $d = \lceil\sqrt{W^2 + H^2}\rceil$ is the image diagonal.

For each pixel $(x_i, y_i)$ and each discretised angle $\theta_j$:

$$
\rho_{ij} = x_i \cos\theta_j + y_i \sin\theta_j
$$

This is computed **fully vectorised** using broadcasting:

```
rho_vals[i, j] = x[i] * cos(θ[j]) + y[i] * sin(θ[j])
```

The accumulator cell $A[\rho_{ij}, \theta_j]$ is incremented for every vote. Peaks in $A$ (cells $\geq$ threshold 50) correspond to the dominant lines.

### Edge Pairing Logic

A thin rod in the image produces **two parallel edge lines** in the skeleton. Rather than just taking the strongest line, we search for a valid second edge:

**1. Wrap-around correction** — Hough space has a $\pi$-periodicity ambiguity. If two lines have angles differing by more than $90°$, we apply:

$$
\theta \leftarrow \theta - \pi, \quad \rho \leftarrow -\rho
$$

to bring them into the same half-plane before comparing.

**2. Physical separation check** — Project both lines to the horizontal midpoint $y_{\text{mid}} = H/4$:

$$
x = \frac{\rho - y_{\text{mid}} \cdot \sin\theta}{\cos\theta}
$$

A candidate line is accepted as the **opposite edge** if:

$$
20 \leq |x_1 - x_2| \leq 60 \quad \text{(pixels)}
$$

**3. Parallelism check:**

$$
|\theta_1 - \theta_2| < 7° = \frac{7\pi}{180}
$$

### Medial Axis Reconstruction

Once both edges are confirmed, the centreline $\rho_{\text{axis}}$ is computed as the true physical midpoint between the two lines at $y_{\text{mid}} = H/2$:

$$
\rho_{\text{axis}} = \rho_1 - \frac{x_1 - x_2}{2} \cdot \cos\theta_1
$$

This corrects for the fact that the midpoint in $\rho$-space is not the same as the midpoint in Cartesian space when lines are not vertical.

The final centreline is drawn by converting back from polar to Cartesian:

$$
(x_0, y_0) = (\rho\cos\theta,\; \rho\sin\theta)
$$

$$
\text{pt}_1 = (x_0 - t\sin\theta,\; y_0 + t\cos\theta), \quad
\text{pt}_2 = (x_0 + t\sin\theta,\; y_0 - t\cos\theta)
$$

for $t = 2 \max(W, H)$ (extends line to image borders).

---

## Output Files

| File | Content |
|---|---|
| `3_0_color_diff.mp4` | Raw BGR background-subtracted difference |
| `3_2_morph_cleaned.mp4` | Binary mask after morphological cleaning |
| `3_3_medial_axis.mp4` | Skeletonised mask (1-pixel wide) |
| `3_4_hough_axis.mp4` | Original frames with detected centreline overlaid in red |
| `Median Background.jpg` | The computed median background model |

---

## Dependencies

```bash
pip install opencv-python numpy
```

---

## Parameters to Tune

| Parameter | Location | Effect |
|---|---|---|
| `bg_diff_threshold = 30` | Stage 1 | Sensitivity of foreground detection. Lower → noisier mask. |
| Kernel size `(5,5)` | Stage 2 | Larger = more aggressive noise removal and hole filling |
| `threshold = 50` | Stage 4 | Minimum Hough votes for a line to be accepted |
| `min_separation = 20` | Stage 4 | Minimum pixel distance between the two detected edges |
| Max separation `60` | Stage 4 | Maximum pixel distance — filters out unrelated lines |
| Parallelism `7°` | Stage 4 | Angular tolerance for the two edges to be considered parallel |
