---
type: article
title: "AR Projections & MediaPipe Pipelines"
description: "Theoretical frameworks of real-time augmented reality projections, homography, HSV color segmentation, and MediaPipe hand landmark tracking."
resource: "opencv-studio"
tags: ["mediapipe", "aruco", "augmented-reality", "homography"]
timestamp: 2026-08-07T14:10:00Z
trust:
  provenance: "internal-wiki"
  freshness: "v0.2"
  lifecycle: "active"
---

# AR Projections & MediaPipe Landmarks

This document covers the advanced interaction pipelines implemented in the **AR Launchpad**, detailing facial Visors, ArUco pillar projections, HSV segmentation (Invisibility Cloak), and gesture-controlled interfaces.

---

## 1. Homography Mapping & ArUco Projections
To project virtual 3D structures (like holograms or pillars) onto flat physical markers in a camera feed, the relative spatial transformation between the marker plane and the camera plane must be calculated.

### Mathematical Homography:
A homography matrix $H$ is a $3 \times 3$ transformation mapping coordinates from a source plane $(x_i, y_i)$ to a destination plane $(x'_i, y'_i)$:
$$\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} \approx H \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$
Since $H$ is defined up to a scale factor, it has 8 degrees of freedom. It is computed from four matching points using the **Direct Linear Transform (DLT)** algorithm.

### 3D Projection Pipeline:
1.  **ArUco Marker Detection:** The corners of a predefined square marker are extracted with sub-pixel precision.
2.  **Pose Estimation (`cv2.solvePnP`):** Resolves the rotation vector ($\vec{r}$) and translation vector ($\vec{t}$) of the marker relative to the camera focal center:
    $$P_{\text{camera}} = R \cdot P_{\text{object}} + T$$
3.  **Projecting 3D Coordinates (`cv2.projectPoints`):** Projects 3D points representing pillars or cubes onto the 2D camera viewport using pinhole camera intrinsic parameters (focal length $f_x, f_y$ and optical center $c_x, c_y$).

---

## 2. Invisibility Cloak: HSV Color Segmentation
The "Invisibility Cloak" filter works by replacing pixels of a specific target color (e.g., green or blue) with a pre-recorded background frame.

### HSV vs. BGR Color Spaces:
Traditional BGR (Blue, Green, Red) encodes colors using raw channel intensity. BGR is highly sensitive to lighting changes and shadows.
**HSV (Hue, Saturation, Value)** separates color frequency from brightness:
*   **Hue (H):** Represents the pure color frequency (0 to 180 in OpenCV).
*   **Saturation (S):** Represents color intensity or purity (0 to 255).
*   **Value (V):** Represents brightness (0 to 255).

### Cloaking Pipeline:
1.  **Calibration:** Capture a static frame of the empty background scene.
2.  **Segmentation Mask:** Detect the target green color within a specific HSV range:
    $$\text{Mask} = (H_{\min} \le H \le H_{\max}) \land (S_{\min} \le S \le S_{\max}) \land (V_{\min} \le V \le V_{\max})$$
3.  **Dilation & Smoothing:** Dilate the mask with a kernel to close gaps and apply a Gaussian blur to soften the borders.
4.  **Blending:** Replace mask regions with background pixels, and keep original pixels elsewhere:
    $$\text{Frame}_{\text{out}} = \text{Frame}_{\text{orig}} \cdot (1 - \text{Mask}) + \text{Background} \cdot \text{Mask}$$

---

## 3. Hand Gesture Recognition & MediaPipe
MediaPipe Hand Tracking runs a lightweight ML pipeline yielding 21 3D hand coordinates.

### Gesture Cycling Control:
To cycle filters in real-time, the relative coordinates of the index finger tip, thumb, and palm wrist are evaluated:
1.  **Swipe Right Gesture:** Triggered when the index finger tip ($X_{8}$) moves significantly to the right of the wrist ($X_{0}$) across consecutive frames.
2.  **Swipe Left Gesture:** Triggered when the index finger tip ($X_{8}$) moves significantly to the left of the wrist ($X_{0}$) across consecutive frames.
3.  **Pinch/Zoom Gestures:** Computed by calculating the Euclidean distance between the thumb tip ($X_{4}$) and the index finger tip ($X_{8}$). If distance is below a threshold, a "pinch select" is registered.
