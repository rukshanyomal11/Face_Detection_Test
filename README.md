# Face Detection Web App

## Install
pip install -r requirements.txt

## Run
uvicorn web.app:app --reload

Open browser:
http://127.0.0.1:8000

Notes:
- MediaPipe does not support Python 3.13 yet. On 3.13 the app falls back to OpenCV Haar detection.
- For best accuracy, use Python 3.10 or 3.11.

Storage cleanup (outputs):
- OUTPUT_MAX_AGE_HOURS (default 168)
- OUTPUT_MAX_FILES (default 300)
- OUTPUT_MAX_MB (default 500)
