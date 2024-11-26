# Real-Time Camera Calibration and Perspective Transformation

This project combines camera calibration and real-time video processing for feature detection, matching, and homography-based perspective transformation.

---

## Features
- **Camera Calibration**:
  - Uses chessboard images to calculate the camera matrix and distortion coefficients.
- **Real-Time Feature Matching**:
  - Detects and matches features between a template image and live video feed.
- **Perspective Transformation**:
  - Estimates the homography matrix and overlays a bounding box on the detected object.

---

## Requirements
- Python 3.8+
- OpenCV 4.8+
- NumPy 1.26+

---

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/camera_calibration_homography.git
   cd camera_calibration_homography
   
2. Install dependencies:
   ``` bash
   pip install -r requirements.txt
   
3. Perform camera calibration:
   ``` bash
   python calibration/camera_calibration.py
   
4. Start real-time processing:
   ``` bash
   python real_time_homography.py
