---
type: article
title: "Traditional Computer Vision Fundamentals"
description: "Core algorithms and mathematical principles of traditional computer vision, including Canny edge detection, Harris corner detection, and image resizing interpolation methods."
resource: "opencv-studio"
tags: ["opencv", "computer-vision", "algorithms"]
timestamp: 2026-08-07T14:10:00Z
trust:
  provenance: "internal-wiki"
  freshness: "v0.2"
  lifecycle: "active"
---

# Traditional Computer Vision Fundamentals

While deep learning (like YOLO) specializes in semantic object recognition, traditional computer vision algorithms remain essential for low-level pixel manipulation, feature extraction, and high-performance real-time preprocessing.

---

## 1. Edge Detection: Canny Algorithm
Canny edge detection is a multi-stage pipeline designed to extract structural borders while suppressing noise.

### The Five Stages of Canny:
1.  **Gaussian Noise Reduction:** A Gaussian filter is applied to smooth the image and remove high-frequency noise.
2.  **Intensity Gradient Calculation:** Sobel kernels ($G_x$ and $G_y$) compute the horizontal and vertical spatial derivatives:
    $$G = \sqrt{G_x^2 + G_y^2}, \quad \theta = \arctan\left(\frac{G_y}{G_x}\right)$$
3.  **Non-Maximum Suppression (NMS):** Thins the thick edges by checking if the gradient magnitude at a pixel is a local maximum along the gradient direction $\theta$. If not, it is suppressed to zero.
4.  **Double Thresholding:** Classifies pixels into strong, weak, and non-edges using two thresholds:
    *   $\text{Value} \ge \text{High Threshold} \implies$ Strong edge.
    *   $\text{Low Threshold} \le \text{Value} < \text{High Threshold} \implies$ Weak edge.
    *   $\text{Value} < \text{Low Threshold} \implies$ Suppressed.
5.  **Hysteresis Edge Tracking:** Weak edges are preserved only if they are spatially connected to a strong edge. This prevents broken segments.

---

## 2. Feature Point Extraction: Harris Corners
Harris Corner Detection identifies points of interest (corners) by analyzing local intensity shifts under small windows.

### Mathematical Formulation:
The change in intensity for a shift $[u,v]$ is approximated by:
$$E(u,v) \approx [u, v] M \begin{bmatrix} u \\ v \end{bmatrix}$$
Where $M$ is the structure tensor (second-moment matrix):
$$M = \sum_{x,y} w(x,y) \begin{bmatrix} I_x^2 & I_x I_y \\ I_x I_y & I_y^2 \end{bmatrix}$$
A corner is characterized by large intensity shifts in all directions, meaning both eigenvalues ($\lambda_1, \lambda_2$) of $M$ are large.

The Harris response score $R$ is calculated as:
$$R = \det(M) - k \cdot (\text{trace}(M))^2$$
Where:
*   $\det(M) = \lambda_1 \lambda_2$
*   $\text{trace}(M) = \lambda_1 + \lambda_2$
*   $k$ is an empirical constant (usually $0.04 - 0.06$).

*   **Corner:** $R > 0$ (both $\lambda_1, \lambda_2$ are large).
*   **Edge:** $R < 0$ (one eigenvalue is much larger than the other).
*   **Flat Region:** $|R|$ is small (both eigenvalues are near zero).

---

## 3. Resizing & Interpolation Methods
When scaling images, new pixel coordinates must be mapped back to source fractional coordinates. The mapping quality is determined by the interpolation method:

*   **Nearest-Neighbor (`cv2.INTER_NEAREST`):** Maps the target pixel to the closest source pixel.
    *   *Tradeoff:* Fast, but introduces severe aliasing and jagged edges.
*   **Bilinear Interpolation (`cv2.INTER_LINEAR`):** Computes pixel values based on a weighted average of the four closest pixels.
    *   *Tradeoff:* Balanced speed and smoothness. Standard default for general image downscaling.
*   **Bicubic Interpolation (`cv2.INTER_CUBIC`):** Uses a weighted average of the $4 \times 4$ surrounding pixels (16 total).
    *   *Tradeoff:* Slower, but yields sharper, visually cleaner upscaled frames.
*   **Lanczos Interpolation (`cv2.INTER_LANCZOS4`):** Uses a sinc filter over an $8 \times 8$ pixel neighborhood.
    *   *Tradeoff:* The highest visual quality and sharpness, but computationally expensive.
