import cv2
import numpy as np
from PIL import Image

# Import advanced face detection libraries
try:
    import mediapipe as mp
except Exception:
    mp = None

try:
    from mtcnn import MTCNN
except Exception:
    MTCNN = None

try:
    import face_recognition
except Exception:
    face_recognition = None

# Check MediaPipe availability
_HAS_MEDIAPIPE = bool(mp) and hasattr(mp, "solutions") and hasattr(mp.solutions, "face_detection")

# Initialize MediaPipe
if _HAS_MEDIAPIPE:
    mp_face = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
else:
    mp_face = None
    mp_drawing = None

# Initialize MTCNN
_HAS_MTCNN = bool(MTCNN)
if _HAS_MTCNN:
    try:
        _MTCNN_DETECTOR = MTCNN()
    except Exception:
        _HAS_MTCNN = False
        _MTCNN_DETECTOR = None
else:
    _MTCNN_DETECTOR = None

# Check face_recognition availability
_HAS_FACE_RECOGNITION = bool(face_recognition)

# Haar cascade fallback
_HAAR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_HAAR = cv2.CascadeClassifier(_HAAR_PATH)
if _HAAR.empty():
    raise RuntimeError(f"Could not load Haar cascade from {_HAAR_PATH}")

def preprocess_image(bgr_image):
    """
    Enhance image quality for better face detection accuracy
    """
    # Convert to float for processing
    image = bgr_image.astype(np.float32) / 255.0
    
    # Denoise the image
    denoised = cv2.bilateralFilter((image * 255).astype(np.uint8), 9, 75, 75)
    
    # Improve contrast using CLAHE
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    # Resize if image is too large (for performance)
    height, width = enhanced.shape[:2]
    max_dimension = 1920
    if width > max_dimension or height > max_dimension:
        scale = max_dimension / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        enhanced = cv2.resize(enhanced, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    
    return enhanced


def detect_faces_mediapipe(bgr_image, min_conf=0.5):
    """
    Detect faces using MediaPipe
    """
    faces = []
    if not _HAS_MEDIAPIPE:
        return faces
    
    h, w, _ = bgr_image.shape
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    
    with mp_face.FaceDetection(model_selection=0, min_detection_confidence=min_conf) as face_detection:
        results = face_detection.process(rgb)
    
    if results.detections:
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            x = max(0, int(bbox.xmin * w))
            y = max(0, int(bbox.ymin * h))
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)
            
            x2 = min(w - 1, x + bw)
            y2 = min(h - 1, y + bh)
            
            confidence = det.score[0]
            faces.append({
                'bbox': (x, y, x2, y2),
                'confidence': confidence,
                'method': 'MediaPipe'
            })
    
    return faces


def detect_faces_mtcnn(bgr_image, min_conf=0.5):
    """
    Detect faces using MTCNN
    """
    faces = []
    if not _HAS_MTCNN or _MTCNN_DETECTOR is None:
        return faces
    
    try:
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        detections = _MTCNN_DETECTOR.detect_faces(rgb)
        
        for det in detections:
            if det['confidence'] >= min_conf:
                x, y, width, height = det['box']
                x = max(0, x)
                y = max(0, y)
                x2 = min(bgr_image.shape[1] - 1, x + width)
                y2 = min(bgr_image.shape[0] - 1, y + height)
                
                faces.append({
                    'bbox': (x, y, x2, y2),
                    'confidence': det['confidence'],
                    'method': 'MTCNN'
                })
    except Exception as e:
        print(f"MTCNN detection error: {e}")
    
    return faces


def detect_faces_face_recognition(bgr_image):
    """
    Detect faces using face_recognition library
    """
    faces = []
    if not _HAS_FACE_RECOGNITION:
        return faces
    
    try:
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb, model='hog')
        
        for (top, right, bottom, left) in face_locations:
            faces.append({
                'bbox': (left, top, right, bottom),
                'confidence': 0.8,  # face_recognition doesn't provide confidence scores
                'method': 'face_recognition'
            })
    except Exception as e:
        print(f"face_recognition detection error: {e}")
    
    return faces


def detect_faces_haar(bgr_image, min_conf=0.3):
    """
    Detect faces using Haar Cascades (fallback method)
    """
    faces = []
    try:
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        face_rects = _HAAR.detectMultiScale(
            gray, 
            scaleFactor=1.05, 
            minNeighbors=5, 
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        for (x, y, w, h) in face_rects:
            faces.append({
                'bbox': (x, y, x + w, y + h),
                'confidence': 0.7,  # Haar doesn't provide confidence, use fixed value
                'method': 'Haar'
            })
    except Exception as e:
        print(f"Haar cascade detection error: {e}")
    
    return faces


def calculate_overlap(bbox1, bbox2):
    """
    Calculate IoU (Intersection over Union) between two bounding boxes
    """
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    # Calculate intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # Calculate union
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def merge_detections(all_faces, overlap_threshold=0.3):
    """
    Merge overlapping face detections from different methods
    """
    if not all_faces:
        return []
    
    # Sort by confidence
    all_faces.sort(key=lambda x: x['confidence'], reverse=True)
    
    merged_faces = []
    used_indices = set()
    
    for i, face in enumerate(all_faces):
        if i in used_indices:
            continue
        
        # Find all overlapping faces
        overlapping = [face]
        for j, other_face in enumerate(all_faces[i+1:], i+1):
            if j in used_indices:
                continue
            
            overlap = calculate_overlap(face['bbox'], other_face['bbox'])
            if overlap > overlap_threshold:
                overlapping.append(other_face)
                used_indices.add(j)
        
        # Use the face with highest confidence
        best_face = max(overlapping, key=lambda x: x['confidence'])
        merged_faces.append(best_face)
        used_indices.add(i)
    
    return merged_faces


def detect_and_draw_faces(bgr_image, min_conf=0.5):
    """
    Enhanced face detection using ensemble of multiple methods
    """
    # Preprocess image for better detection
    processed_image = preprocess_image(bgr_image.copy())
    
    # Apply multiple detection methods
    all_faces = []
    
    # Method 1: MediaPipe (most accurate for modern images)
    mp_faces = detect_faces_mediapipe(processed_image, min_conf)
    all_faces.extend(mp_faces)
    
    # Method 2: MTCNN (very accurate, especially for small faces)
    mtcnn_faces = detect_faces_mtcnn(processed_image, min_conf)
    all_faces.extend(mtcnn_faces)
    
    # Method 3: face_recognition (good for profile faces)
    fr_faces = detect_faces_face_recognition(processed_image)
    all_faces.extend(fr_faces)
    
    # Method 4: Haar Cascades (fallback for challenging conditions)
    if not all_faces:  # Only use Haar if no other method found faces
        haar_faces = detect_faces_haar(processed_image, min_conf * 0.7)
        all_faces.extend(haar_faces)
    
    # Merge overlapping detections
    final_faces = merge_detections(all_faces, overlap_threshold=0.3)
    
    # Draw bounding boxes on original image
    output_image = bgr_image.copy()
    face_count = len(final_faces)
    
    for i, face in enumerate(final_faces):
        x1, y1, x2, y2 = face['bbox']
        confidence = face['confidence']
        method = face['method']
        
        # Use different colors for different methods
        color_map = {
            'MediaPipe': (0, 255, 0),    # Green
            'MTCNN': (255, 0, 0),        # Blue
            'face_recognition': (0, 165, 255),  # Orange
            'Haar': (255, 255, 0)        # Cyan
        }
        color = color_map.get(method, (0, 255, 0))
        
        # Draw rectangle
        cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)
        
        # Add label
        label = f"Face {i+1} ({method}: {confidence:.2f})"
        label_y = max(20, y1 - 10)
        cv2.putText(
            output_image, 
            label,
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6,
            color, 
            2
        )
    
    return output_image, face_count
