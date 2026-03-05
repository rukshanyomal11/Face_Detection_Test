import os
import sys
import uuid
import cv2
import json
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


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "output_url": None,
        "face_count": None,
        "error": None
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
        
        # Generate analysis report
        analysis_report = generate_face_analysis_report(face_details)

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
            "error": None
        })

    except Exception as e:
        error_msg = f"Processing error: {str(e)}"
        print(f"Error processing image: {e}")  # Log for debugging
        
        return templates.TemplateResponse("index.html", {
            "request": request,
            "output_url": None,
            "face_count": None,
            "error": error_msg
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
        
        # Clean up temp files
        for temp_path in temp_paths:
            try:
                os.remove(temp_path)
            except:
                pass
        
        return JSONResponse(content={
            "success": True,
            "total_processed": len(results),
            "results": results
        })
        
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
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )
