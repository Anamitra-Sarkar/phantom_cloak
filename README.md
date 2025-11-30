# PHANTOM-CLOAK

Real-Time Optical Camouflage System using Python and OpenCV.

## Overview

PHANTOM-CLOAK is a computer vision application that creates an "invisibility cloak" effect by replacing the user's body pixels with a pre-captured background. The system uses MediaPipe for human segmentation and applies advanced VFX techniques including a "light bending" distortion effect at the edges for a realistic camouflage experience.

## Features

- **Absolute Invisibility Mode**: Clean background replacement with seamless blending
- **Predator Shimmer Mode**: Light-bending distortion effect with chromatic aberration
- **Military Prototype HUD**: Real-time status display with FPS counter
- **Live Recalibration**: Recapture background without restarting
- **Adjustable Refraction Index**: Control the intensity of the shimmer effect

## Requirements

- Python 3.10+
- Webcam

## Installation

```bash
# Clone the repository
git clone https://github.com/Anamitra-Sarkar/phantom_cloak.git
cd phantom_cloak

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run with default camera (ID: 0)
python main.py

# Run with specific camera
python main.py 1
```

## Controls

| Key | Action |
|-----|--------|
| `C` | Recalibrate background |
| `M` | Switch mode (ABSOLUTE/PREDATOR) |
| `+` | Increase refraction index |
| `-` | Decrease refraction index |
| `Q` | Quit application |

## How It Works

### 1. Calibration Phase
On startup, the system captures 30 frames of the static background and averages them to create a noise-reduced `background_plate`.

### 2. Masking Engine
Uses MediaPipe's Selfie Segmentation to extract the human silhouette. The mask is refined with Gaussian blur for seamless edge blending.

### 3. Invisibility Modes

**Mode 1: Absolute Invisibility**
- Replaces pixels where the mask exceeds the threshold with background pixels
- Uses soft edge blending for natural transitions

**Mode 2: Predator Shimmer**
- Detects edges of the mask using Canny edge detection
- Applies displacement mapping (`cv2.remap`) to simulate light bending
- Adds chromatic aberration for realistic optical distortion
- Animated shimmer effect based on time

## Architecture

```
phantom_cloak/
├── main.py           # Main application loop
├── vfx_utils.py      # VFX math and distortion effects
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

## Performance

The system is optimized to run at 30+ FPS on CPU by:
- Using MediaPipe's optimized segmentation model
- Processing at 640x480 resolution
- Efficient NumPy operations for pixel manipulation

## License

Apache License 2.0
