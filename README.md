# OpenCV Stereo Camera Calibration and Depth Mapping

A comprehensive implementation of stereo camera calibration, depth map generation, and 3D point cloud visualization using OpenCV and Python. This project demonstrates both calibrated and uncalibrated stereo vision techniques for depth estimation and object distance measurement.

## Overview

This project implements stereo vision algorithms to:
- Calibrate stereo camera systems using chessboard patterns
- Generate depth maps from stereo image pairs
- Create 3D point clouds exportable to PLY format for MeshLab visualization  
- Calculate real-world distances to objects with proximity warnings
- Handle both calibrated stereo camera setups and single camera uncalibrated approaches

## Features

- **Stereo Camera Calibration**: Automatic calibration using chessboard corner detection
- **Depth Map Generation**: Semi-Global Block Matching (SGBM) algorithm with WLS filtering
- **Point Cloud Export**: PLY file generation for 3D visualization in MeshLab
- **Distance Measurement**: Real-time object distance calculation with proximity alerts
- **Uncalibrated Stereo**: Depth estimation from single camera at different positions
- **Interactive Interface**: Mouse-click distance measurement on disparity maps

## Project Structure

```
├── REPORT.ipynb              # Detailed theoretical analysis and implementation
├── demo_distance.py          # Real-time distance measurement from stereo cameras
├── demo_pcloud.py           # Point cloud generation and PLY export
├── demo_uncalibCam.py       # Uncalibrated stereo depth mapping
├── getImageStereo.py        # Stereo image capture utility
├── uncalibCam.py            # Alternative uncalibrated stereo implementation
├── images/                  # Calibration and test images
│   ├── stereoLeft/          # Left camera calibration images (49 images)
│   └── stereoRight/         # Right camera calibration images (49 images)
├── reportImage/             # Documentation images
├── point_cloud.ply          # Generated 3D point cloud
├── left_image.png           # Sample left stereo image
├── right_image.png          # Sample right stereo image
├── unCalibLeft.png          # Uncalibrated left image
├── unCalibRight.png         # Uncalibrated right image
└── opencv-env/              # Python virtual environment
```

## Technical Implementation

### Stereo Camera Calibration
- Uses 9x6 chessboard pattern for calibration
- Implements corner detection with sub-pixel accuracy
- Performs stereo rectification for epipolar line alignment
- Computes camera intrinsic and extrinsic parameters

### Depth Map Generation
- **Algorithm**: Semi-Global Block Matching (StereoSGBM)
- **Filtering**: Weighted Least Squares (WLS) filter for noise reduction
- **Post-processing**: Morphological closing for hole filling
- **Visualization**: Ocean colormap for enhanced depth perception

### Distance Calculation
The distance estimation uses an experimentally derived polynomial regression:
```
Distance = -593.97 * disparity³ + 1506.8 * disparity² - 1373.1 * disparity + 522.06
```
Proximity warning triggered when objects are closer than 50cm.

## Requirements

```
numpy
opencv-python
opencv-contrib-python
matplotlib
plyfile
glob
IPython
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/lanhp-vn/opencv-stereo-camera-calibration-MeshLab.git
cd opencv-stereo-camera-calibration-MeshLab
```

2. Create and activate virtual environment:
```bash
python -m venv opencv-env
# Windows
opencv-env\Scripts\activate
# Linux/Mac
source opencv-env/bin/activate
```

3. Install dependencies:
```bash
pip install numpy opencv-python opencv-contrib-python matplotlib plyfile
```

## Usage

### Stereo Camera Calibration and Distance Measurement
```bash
python demo_distance.py
```
- Calibrates stereo cameras using provided chessboard images
- Opens real-time camera feed with distance measurement
- Double-click on objects to measure distance
- Press space to exit

### Point Cloud Generation
```bash
python demo_pcloud.py
```
- Processes stereo images to generate depth map
- Exports 3D point cloud to `point_cloud.ply`
- Visualize the PLY file in MeshLab or similar 3D software

### Uncalibrated Stereo Processing
```bash
python demo_uncalibCam.py
```
- Generates depth maps from single camera images taken at different positions
- Uses the same calibration parameters for rectification

### Image Capture Utility
```bash
python getImageStereo.py
```
- Captures synchronized stereo image pairs
- Press 's' to save images, ESC to exit

## Theoretical Background

The project implements epipolar geometry principles for stereo vision:

1. **Triangulation**: 3D point reconstruction from corresponding 2D points
2. **Epipolar Constraint**: Reduces correspondence search to 1D along epipolar lines
3. **Essential Matrix**: Encodes geometric relationship between stereo views
4. **Mathematical Proof**: Demonstrates that corresponding points lie on horizontal lines for parallel camera setup

## Results

- **Disparity Maps**: High-quality depth maps with WLS filtering
- **Point Clouds**: Detailed 3D reconstructions exportable to MeshLab
- **Distance Accuracy**: Real-time distance measurement with polynomial regression
- **Proximity Detection**: Automatic warnings for objects within 50cm
