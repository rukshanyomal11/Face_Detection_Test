# Face Detection Web App

A FastAPI-based face detection studio with a polished web interface for:

- single-image face detection
- advanced face analysis
- batch image processing
- face comparison
- live webcam detection
- JSON and CSV report export

The app combines multiple computer vision libraries and automatically falls back to lighter detectors when some models are unavailable.

## Highlights

- Modern browser UI built with FastAPI and Jinja templates
- Multiple detection pipelines including MediaPipe, MTCNN, face_recognition, and OpenCV Haar cascades
- Analysis modes for basic detection, advanced attributes, landmarks, and full analysis
- Batch processing for up to 10 images at once
- Face comparison endpoint for similarity checking
- Live webcam tab with `Fast` and `Accurate` modes
- Automatic cleanup for generated output files
- Downloadable processed images plus JSON and CSV exports for reports

## Feature Overview

### 1. Single Image Analysis

Upload one `JPG` or `PNG` image and choose an analysis mode:

- `basic`
  Detect faces and draw bounding boxes
- `advanced`
  Detect faces and attempt age, gender, emotion, and race analysis
- `landmarks`
  Detect facial landmarks
- `full`
  Combine detection, landmarks, and advanced analysis

### 2. Batch Processing

- Upload up to `10` images in one request
- Run a shared analysis mode across all selected files
- Get per-image detection results and processed outputs

### 3. Face Comparison

- Upload two images
- Compare detected faces
- Return similarity, distance, and same-person estimate

### 4. Live Webcam

Two webcam modes are available:

- `Fast`
  Lower latency, smaller frames, lighter realtime detection
- `Accurate`
  Slower, but uses the heavier detection pipeline for better face pickup

### 5. Report and Output Tools

After a successful analysis, the UI can show:

- total detected faces
- average age and distributions when available
- detection method breakdown
- confidence histogram
- face-size histogram
- individual face cards with crops
- processed image download
- JSON export
- CSV export

## Tech Stack

- `FastAPI`
- `Uvicorn`
- `Jinja2`
- `OpenCV`
- `MediaPipe`
- `MTCNN`
- `face_recognition`
- `DeepFace`
- `NumPy`

## Project Structure

```text
face-detection-web-app/
|-- README.md
|-- requirements.txt
|-- src/
|   `-- utils/
|       `-- detector.py
`-- web/
    |-- app.py
    |-- static/
    |   `-- outputs/
    `-- templates/
        |-- index.html
        `-- partials/
```

## Requirements

Recommended:

- Python `3.10` or `3.11`
- Windows, Linux, or macOS with webcam/browser support

Important version note:

- `MediaPipe` does not support Python `3.13` in this project setup.
- On Python `3.13`, the app falls back more often to OpenCV Haar-based detection.
- The app still runs on Python `3.13`, but detection quality can be lower, especially for webcam use.

## Installation

### Windows PowerShell

Recommended setup with Python 3.11:

```powershell
cd C:\Users\Rukshan\Desktop\heee\face-detection-web-app
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you already have the virtual environment, just activate it and install:

```powershell
cd C:\Users\Rukshan\Desktop\heee\face-detection-web-app
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Running the App

From the project root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn web.app:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

FastAPI also exposes API docs at:

```text
http://127.0.0.1:8000/docs
http://127.0.0.1:8000/redoc
```

## How to Use

### Single Image

1. Open the app in your browser.
2. Stay on the `Single Image` tab.
3. Upload a `JPG` or `PNG`.
4. Choose an analysis mode.
5. Adjust confidence if needed.
6. Click `Analyze Image`.

### Batch Processing

1. Open the `Batch Processing` tab.
2. Select up to `10` images.
3. Choose the analysis mode.
4. Click `Process Batch`.

### Face Comparison

1. Open the `Face Comparison` tab.
2. Upload two face images.
3. Click `Compare Faces`.

### Live Webcam

1. Open the `Live Webcam` tab.
2. Choose a webcam mode:
   - `Fast` for smoother updates
   - `Accurate` for stronger detection
3. Adjust detection confidence.
4. Click `Start Camera`.

## Webcam Mode Guide

### Fast Mode

Best when:

- you want smoother live updates
- you are testing general webcam functionality
- you are on lower-powered hardware

Tradeoff:

- may miss difficult faces
- weaker with blur, side angles, occlusion, or poor light

### Accurate Mode

Best when:

- the webcam is missing your face in fast mode
- your face is farther from the camera
- lighting is not ideal
- you want higher detection quality and accept more delay

Tradeoff:

- slower frame processing
- higher CPU usage

## API Endpoints

### `GET /`

Main web interface.

### `POST /upload`

HTML form endpoint for single-image analysis.

Form fields:

- `file`
- `analysis_mode`
- `min_confidence`

### `POST /batch-upload`

Process multiple images.

Form fields:

- `files`
- `analysis_mode`
- `min_confidence`

Notes:

- maximum `10` images per batch

### `POST /compare-faces`

Compare two uploaded images.

Form fields:

- `file1`
- `file2`

### `POST /api/detect`

JSON API for face detection.

Form fields:

- `file`
- `analysis_mode`
- `min_confidence`
- `return_crops`

Typical response includes:

- `success`
- `face_count`
- `faces`
- processed image URL
- optional face crops
- analysis report data

### `POST /api/webcam-detect`

Realtime webcam frame API.

Form fields:

- `file`
- `min_confidence`
- `webcam_mode`
- `return_frame`

Typical response includes:

- `success`
- `face_count`
- `mode`
- `faces`
- `detector`
- `processing_ms`
- `frame_size`
- optional `frame` base64 image

### `GET /api/analysis/{image_id}`

Placeholder endpoint for future persistent analysis storage.

## Output Storage and Cleanup

Processed images are stored in:

```text
web/static/outputs/
```

The app automatically cleans old files during startup and processing.

Environment variables:

- `OUTPUT_MAX_AGE_HOURS`
  Default: `168`
- `OUTPUT_MAX_FILES`
  Default: `300`
- `OUTPUT_MAX_MB`
  Default: `500`

Example:

```powershell
$env:OUTPUT_MAX_AGE_HOURS="72"
$env:OUTPUT_MAX_FILES="200"
$env:OUTPUT_MAX_MB="250"
.\.venv\Scripts\python.exe -m uvicorn web.app:app --reload
```

## Troubleshooting

### Webcam does not detect a face

Try this first:

- switch from `Fast` to `Accurate`
- face the camera directly
- improve front lighting
- avoid covering part of your face
- move closer to the camera
- lower the confidence slider slightly

### Webcam is slow

Try this:

- use `Fast` mode
- reduce background CPU load
- use Python `3.10` or `3.11`
- close other heavy apps using the camera or CPU

### Python 3.13 gives weaker detection

This is expected in this setup because MediaPipe is not installed there. For best results, recreate the environment with Python `3.10` or `3.11`.

### Camera permission error

Make sure:

- your browser has camera permission
- no other app is locking the webcam
- you are opening the app at `http://127.0.0.1:8000`

### Dependency installation fails

Some packages in this project are heavy machine-learning dependencies. If installation fails:

- upgrade `pip`
- use Python `3.10` or `3.11`
- install in a fresh virtual environment

## Notes for Development

- Main app entry: [web/app.py](web/app.py)
- Detection utilities: [src/utils/detector.py](src/utils/detector.py)
- Main template: [web/templates/index.html](web/templates/index.html)

## Current Dependency List

From [requirements.txt](requirements.txt):

- `fastapi`
- `uvicorn`
- `python-multipart`
- `jinja2`
- `opencv-python`
- `mediapipe` for Python versions below `3.13`
- `mtcnn`
- `face-recognition`
- `dlib-binary`
- `tensorflow`
- `numpy`
- `Pillow`
- `deepface`
- `fer2013-lib`
- `age-gender-estimation`
- `scipy`
- `matplotlib`
- `seaborn`
- `requests`

## Summary

This project is a full-featured face detection web app with both UI and API workflows. If you want the best detection quality, use Python `3.10` or `3.11` and the webcam `Accurate` mode when needed.
