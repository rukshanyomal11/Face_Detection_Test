import os
import uuid
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from src.utils.detector import detect_and_draw_faces

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
OUTPUT_DIR = os.path.join(STATIC_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI()

templates = Jinja2Templates(directory=TEMPLATES_DIR)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "output_url": None,
        "face_count": None,
        "error": None
    })


@app.post("/upload", response_class=HTMLResponse)
async def upload_image(request: Request, file: UploadFile = File(...)):
    try:
        if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
            return templates.TemplateResponse("index.html", {
                "request": request,
                "output_url": None,
                "face_count": None,
                "error": "Please upload a JPG or PNG image."
            })

        content = await file.read()
        img_array = np.frombuffer(content, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            return templates.TemplateResponse("index.html", {
                "request": request,
                "output_url": None,
                "face_count": None,
                "error": "Could not read the image."
            })

        out_img, face_count = detect_and_draw_faces(img, min_conf=0.5)

        out_name = f"{uuid.uuid4().hex}.jpg"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        cv2.imwrite(out_path, out_img)

        output_url = f"/static/outputs/{out_name}"

        return templates.TemplateResponse("index.html", {
            "request": request,
            "output_url": output_url,
            "face_count": face_count,
            "error": None
        })

    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "output_url": None,
            "face_count": None,
            "error": str(e)
        })
