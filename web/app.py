import os
import sys
import uuid
import time
import cv2
import json
import base64
import numpy as np
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

# Add the parent directory to Python path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.detector import (
    detect_and_draw_faces, 
    detect_and_draw_faces_realtime,
    process_batch_images, 
    compare_faces,
    extract_face_crops,
    generate_face_analysis_report,
    detect_face_landmarks
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
OUTPUT_DIR = os.path.join(STATIC_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI()

templates = Jinja2Templates(directory=TEMPLATES_DIR)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

def _env_int(name, default):
    try:
        value = int(os.getenv(name, default))
    except (TypeError, ValueError):
        value = int(default)
    return max(value, 0)

OUTPUT_MAX_AGE_HOURS = _env_int("OUTPUT_MAX_AGE_HOURS", 168)  # 7 days
OUTPUT_MAX_FILES = _env_int("OUTPUT_MAX_FILES", 300)
OUTPUT_MAX_MB = _env_int("OUTPUT_MAX_MB", 500)

def _scan_output_files():
    files = []
    try:
        for entry in os.scandir(OUTPUT_DIR):
            if not entry.is_file():
                continue
            try:
                stat = entry.stat()
            except FileNotFoundError:
                continue
            files.append({
                "path": entry.path,
                "mtime": stat.st_mtime,
                "size": stat.st_size
            })
    except FileNotFoundError:
        return []
    return files

def cleanup_output_dir():
    now = time.time()
    files = _scan_output_files()
    deleted_files = 0
    freed_bytes = 0

    if OUTPUT_MAX_AGE_HOURS > 0:
        cutoff = now - (OUTPUT_MAX_AGE_HOURS * 3600)
        remaining = []
        for info in files:
            if info["mtime"] < cutoff:
                try:
                    os.remove(info["path"])
                    deleted_files += 1
                    freed_bytes += info["size"]
                except OSError:
                    remaining.append(info)
            else:
                remaining.append(info)
        files = remaining

    if OUTPUT_MAX_FILES > 0 and len(files) > OUTPUT_MAX_FILES:
        files.sort(key=lambda x: x["mtime"], reverse=True)
        to_delete = files[OUTPUT_MAX_FILES:]
        files = files[:OUTPUT_MAX_FILES]
        for info in to_delete:
            try:
                os.remove(info["path"])
                deleted_files += 1
                freed_bytes += info["size"]
            except OSError:
                continue

    if OUTPUT_MAX_MB > 0:
        max_bytes = OUTPUT_MAX_MB * 1024 * 1024
        total_bytes = sum(info["size"] for info in files)
        if total_bytes > max_bytes:
            files.sort(key=lambda x: x["mtime"])
            for info in files:
                if total_bytes <= max_bytes:
                    break
                try:
                    os.remove(info["path"])
                    deleted_files += 1
                    freed_bytes += info["size"]
                    total_bytes -= info["size"]
                except OSError:
                    continue

    stats = get_output_stats()
    stats["deleted_files"] = deleted_files
    stats["freed_mb"] = round(freed_bytes / (1024 * 1024), 2)
    return stats

def get_output_stats():
    files = _scan_output_files()
    total_bytes = sum(info["size"] for info in files)
    return {
        "total_files": len(files),
        "total_mb": round(total_bytes / (1024 * 1024), 2),
        "max_age_hours": OUTPUT_MAX_AGE_HOURS,
        "max_files": OUTPUT_MAX_FILES,
        "max_mb": OUTPUT_MAX_MB
    }

@app.on_event("startup")
def _startup_cleanup():
    cleanup_output_dir()

def _json_default(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _json_safe(payload):
    return json.loads(json.dumps(payload, default=_json_default))


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "output_url": None,
        "face_count": None,
        "error": None,
        "analysis_report_json": None,
        "face_details_json": None,
        "output_stats": get_output_stats()
    })


@app.post("/upload", response_class=HTMLResponse)
async def upload_image(
    request: Request, 
    file: UploadFile = File(...), 
    analysis_mode: str = Form("basic"),
    min_confidence: float = Form(0.3)
):
    try:
        if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
            return templates.TemplateResponse("index.html", {
                "request": request,
                "output_url": None,
                "face_count": None,
                "error": "Please upload a JPG or PNG image.",
                "analysis_report_json": None,
                "face_details_json": None,
                "output_stats": get_output_stats()
            })

        content = await file.read()
        img_array = np.frombuffer(content, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            return templates.TemplateResponse("index.html", {
                "request": request,
                "output_url": None,
                "face_count": None,
                "error": "Could not read the image.",
                "analysis_report_json": None,
                "face_details_json": None,
                "output_stats": get_output_stats()
            })

        # Enhanced detection with configurable analysis mode
        out_img, face_count, face_details = detect_and_draw_faces(
            img, 
            min_conf=min_confidence, 
            analysis_mode=analysis_mode
        )

        out_name = f"{uuid.uuid4().hex}.jpg"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        
        # Save with high quality
        cv2.imwrite(out_path, out_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

        output_url = f"/static/outputs/{out_name}"
        cleanup_output_dir()
        
        # Generate analysis report
        analysis_report = generate_face_analysis_report(face_details)
        analysis_report_json = json.dumps(analysis_report, default=_json_default) if analysis_report else None
        face_details_json = json.dumps(face_details, default=_json_default) if face_details else None

        # Provide detailed feedback based on analysis mode
        if analysis_mode == "basic":
            detection_message = f"Detected {face_count} face(s) using advanced AI models"
        elif analysis_mode == "advanced":
            detection_message = f"Advanced analysis complete: {face_count} face(s) with age, gender, emotion data"
        elif analysis_mode == "landmarks":
            detection_message = f"Facial landmarks detected for {face_count} face(s)"
        else:  # full
            detection_message = f"Complete analysis: {face_count} face(s) with all features"
        
        # Extract face crops for display
        face_crops = extract_face_crops(img, [{'bbox': face['bbox'], 'confidence': face['confidence'], 'method': face['method']} for face in face_details])
        
        return templates.TemplateResponse("index.html", {
            "request": request,
            "output_url": output_url,
            "face_count": face_count,
            "detection_message": detection_message,
            "analysis_mode": analysis_mode,
            "face_details": face_details,
            "face_crops": face_crops,
            "analysis_report": analysis_report,
            "analysis_report_json": analysis_report_json,
            "face_details_json": face_details_json,
            "error": None,
            "output_stats": get_output_stats()
        })

    except Exception as e:
        error_msg = f"Processing error: {str(e)}"
        print(f"Error processing image: {e}")  # Log for debugging
        
        return templates.TemplateResponse("index.html", {
            "request": request,
            "output_url": None,
            "face_count": None,
            "error": error_msg,
            "analysis_report_json": None,
            "face_details_json": None,
            "output_stats": get_output_stats()
        })


@app.post("/batch-upload", response_class=JSONResponse)
async def batch_upload_images(
    files: List[UploadFile] = File(...),
    analysis_mode: str = Form("basic"),
    min_confidence: float = Form(0.3)
):
    """
    Process multiple images for face detection
    """
    try:
        if len(files) > 10:  # Limit batch size
            raise HTTPException(status_code=400, detail="Maximum 10 images allowed per batch")
        
        # Save uploaded files temporarily
        temp_paths = []
        for file in files:
            if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
                continue
            
            content = await file.read()
            temp_name = f"temp_{uuid.uuid4().hex}.jpg"
            temp_path = os.path.join(OUTPUT_DIR, temp_name)
            
            img_array = np.frombuffer(content, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img is not None:
                cv2.imwrite(temp_path, img)
                temp_paths.append(temp_path)
        
        # Process batch
        results = process_batch_images(temp_paths, min_confidence, analysis_mode, OUTPUT_DIR)
        cleanup_output_dir()
        
        # Clean up temp files
        for temp_path in temp_paths:
            try:
                os.remove(temp_path)
            except:
                pass
        
        return JSONResponse(content=_json_safe({
            "success": True,
            "total_processed": len(results),
            "results": results,
            "output_stats": get_output_stats()
        }))
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )


@app.post("/compare-faces", response_class=JSONResponse)
async def compare_faces_endpoint(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...)
):
    """
    Compare two faces for similarity
    """
    try:
        # Save both images temporarily
        temp_paths = []
        for i, file in enumerate([file1, file2]):
            if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
                raise HTTPException(status_code=400, detail=f"Invalid file type for image {i+1}")
            
            content = await file.read()
            temp_name = f"compare_{i}_{uuid.uuid4().hex}.jpg"
            temp_path = os.path.join(OUTPUT_DIR, temp_name)
            
            img_array = np.frombuffer(content, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img is None:
                raise HTTPException(status_code=400, detail=f"Could not read image {i+1}")
            
            cv2.imwrite(temp_path, img)
            temp_paths.append(temp_path)
        
        # Compare faces
        comparison_result = compare_faces(temp_paths[0], temp_paths[1])
        
        # Clean up temp files
        for temp_path in temp_paths:
            try:
                os.remove(temp_path)
            except:
                pass
        
        return JSONResponse(content={
            "success": True,
            "comparison": comparison_result
        })
        
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )


@app.get("/api/analysis/{image_id}", response_class=JSONResponse)
async def get_analysis_data(image_id: str):
    """
    Get detailed analysis data for a processed image
    """
    try:
        # This is a simplified version - in production you'd store analysis results
        return JSONResponse(content={
            "success": True,
            "message": "Analysis data endpoint - implement persistent storage for production use"
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )


@app.post("/api/detect", response_class=JSONResponse)
async def detect_api(
    file: UploadFile = File(...),
    analysis_mode: str = Form("basic"),
    min_confidence: float = Form(0.3),
    return_crops: bool = Form(False)
):
    """
    JSON API endpoint for face detection
    """
    try:
        if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
            raise HTTPException(status_code=400, detail="Invalid file type. Use JPEG or PNG.")

        content = await file.read()
        img_array = np.frombuffer(content, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image.")

        out_img, face_count, face_details = detect_and_draw_faces(
            img, min_conf=min_confidence, analysis_mode=analysis_mode
        )

        out_name = f"{uuid.uuid4().hex}.jpg"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        cv2.imwrite(out_path, out_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cleanup_output_dir()

        response_data = {
            "success": True,
            "face_count": face_count,
            "output_url": f"/static/outputs/{out_name}",
            "faces": face_details,
            "analysis_report": generate_face_analysis_report(face_details)
        }

        if return_crops:
            response_data["face_crops"] = extract_face_crops(
                img,
                [{"bbox": f["bbox"], "confidence": f["confidence"], "method": f["method"]} for f in face_details]
            )

        return JSONResponse(content=_json_safe(response_data))

    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.post("/api/webcam-detect", response_class=JSONResponse)
async def webcam_detect(
    file: UploadFile = File(...),
    min_confidence: float = Form(0.3),
    webcam_mode: str = Form("fast"),
    return_frame: bool = Form(True)
):
    """
    Process a single webcam frame and return the annotated image as base64
    """
    try:
        start_time = time.perf_counter()
        content = await file.read()
        img_array = np.frombuffer(content, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            return JSONResponse(content={"success": False, "error": "Could not decode frame"})

        webcam_mode = (webcam_mode or "fast").strip().lower()
        if webcam_mode == "accurate":
            out_img, face_count, face_details = detect_and_draw_faces(
                img,
                min_conf=min_confidence,
                analysis_mode="basic"
            )
            methods = sorted({face.get("method") for face in face_details if face.get("method")})
            detector_name = ", ".join(methods) if methods else "Accurate Pipeline"
        else:
            webcam_mode = "fast"
            out_img, face_count, face_details, detector_name = detect_and_draw_faces_realtime(
                img, min_conf=min_confidence
            )

        response_data = {
            "success": True,
            "face_count": face_count,
            "mode": webcam_mode,
            "faces": [
                {"id": f["id"], "bbox": f["bbox"], "confidence": round(f["confidence"], 2), "method": f["method"]}
                for f in face_details
            ],
            "detector": detector_name,
            "processing_ms": round((time.perf_counter() - start_time) * 1000, 1),
            "frame_size": {
                "width": int(out_img.shape[1]),
                "height": int(out_img.shape[0])
            }
        }

        if return_frame:
            _, buffer = cv2.imencode(".jpg", out_img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            response_data["frame"] = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"

        return JSONResponse(content=_json_safe(response_data))

    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.post("/api/detect", response_class=JSONResponse)
async def api_detect_faces(
    file: UploadFile = File(...),
    analysis_mode: str = Form("basic"),
    min_confidence: float = Form(0.3),
    return_crops: bool = Form(False)
):
    """
    API endpoint for face detection that returns JSON data
    """
    try:
        if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
            raise HTTPException(status_code=400, detail="Invalid file type")

        content = await file.read()
        img_array = np.frombuffer(content, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Could not read image")

        # Detect faces
        out_img, face_count, face_details = detect_and_draw_faces(
            img, min_conf=min_confidence, analysis_mode=analysis_mode
        )
        
        # Save processed image
        out_name = f"api_{uuid.uuid4().hex}.jpg"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        cv2.imwrite(out_path, out_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cleanup_output_dir()
        
        result = {
            "success": True,
            "face_count": face_count,
            "faces": face_details,
            "processed_image_url": f"/static/outputs/{out_name}",
            "analysis_mode": analysis_mode
        }
        
        # Add face crops if requested
        if return_crops:
            face_crops = extract_face_crops(img, [
                {'bbox': face['bbox'], 'confidence': face['confidence'], 'method': face['method']} 
                for face in face_details
            ])
            result["face_crops"] = face_crops
        
        return JSONResponse(content=_json_safe(result))
        
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )
