import cv2
import numpy as np
from PIL import Image
import json
import base64
import uuid
import os
from io import BytesIO

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

try:
    from deepface import DeepFace
except Exception:
    DeepFace = None

# Check MediaPipe availability
_HAS_MEDIAPIPE = bool(mp) and hasattr(mp, "solutions") and hasattr(mp.solutions, "face_detection")

# Initialize MediaPipe
if _HAS_MEDIAPIPE:
    mp_face = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing_styles = mp.solutions.drawing_styles
    try:
        _MP_FACE_DETECTOR = mp_face.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.1
        )
    except Exception:
        _MP_FACE_DETECTOR = None
else:
    mp_face = None
    mp_drawing = None
    mp_face_mesh = None
    mp_drawing_styles = None
    _MP_FACE_DETECTOR = None

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

# Check DeepFace availability
_HAS_DEEPFACE = bool(DeepFace)

# Check face_recognition availability
_HAS_FACE_RECOGNITION = bool(face_recognition)

# Haar cascade fallback
_HAAR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_HAAR = cv2.CascadeClassifier(_HAAR_PATH)
if _HAAR.empty():
    raise RuntimeError(f"Could not load Haar cascade from {_HAAR_PATH}")


def _to_python_number(value):
    """
    Convert NumPy scalar values into plain Python numbers for JSON responses.
    """
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _normalize_bbox(bbox):
    return tuple(int(_to_python_number(coord)) for coord in bbox)


def _normalize_face_info(face_info):
    normalized = {}
    for key, value in face_info.items():
        if key == 'bbox' and isinstance(value, (list, tuple)):
            normalized[key] = [int(_to_python_number(coord)) for coord in value]
        elif isinstance(value, np.ndarray):
            normalized[key] = value.tolist()
        else:
            normalized[key] = _to_python_number(value)
    return normalized

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


def resize_for_realtime(bgr_image, max_dimension=480):
    """
    Downscale webcam frames to keep realtime detection responsive.
    """
    height, width = bgr_image.shape[:2]
    largest_dimension = max(width, height)
    if largest_dimension <= max_dimension:
        return bgr_image.copy()

    scale = max_dimension / float(largest_dimension)
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))
    return cv2.resize(bgr_image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def detect_faces_mediapipe(bgr_image, min_conf=0.5):
    """
    Detect faces using MediaPipe
    """
    faces = []
    if not _HAS_MEDIAPIPE:
        return faces
    
    h, w, _ = bgr_image.shape
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    
    if _MP_FACE_DETECTOR is None:
        return faces

    results = _MP_FACE_DETECTOR.process(rgb)
    
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
            if confidence < min_conf:
                continue
            faces.append({
                'bbox': _normalize_bbox((x, y, x2, y2)),
                'confidence': float(_to_python_number(confidence)),
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
                    'bbox': _normalize_bbox((x, y, x2, y2)),
                    'confidence': float(_to_python_number(det['confidence'])),
                    'method': 'MTCNN'
                })
    except Exception as e:
        print(f"MTCNN detection error: {e}")
    
    return faces


def detect_faces_face_recognition(bgr_image, min_conf=0.5):
    """
    Detect faces using face_recognition library
    """
    faces = []
    if not _HAS_FACE_RECOGNITION:
        return faces
    
    try:
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb, model='hog')
        
        fixed_confidence = 0.8
        if fixed_confidence < min_conf:
            return faces

        for (top, right, bottom, left) in face_locations:
            faces.append({
                'bbox': _normalize_bbox((left, top, right, bottom)),
                'confidence': fixed_confidence,  # face_recognition doesn't provide confidence scores
                'method': 'face_recognition'
            })
    except Exception as e:
        print(f"face_recognition detection error: {e}")
    
    return faces


def detect_faces_haar(
    bgr_image,
    min_conf=0.3,
    scale_factor=1.05,
    min_neighbors=5,
    min_size=(30, 30)
):
    """
    Detect faces using Haar Cascades (fallback method)
    """
    faces = []
    try:
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        face_rects = _HAAR.detectMultiScale(
            gray, 
            scaleFactor=scale_factor, 
            minNeighbors=min_neighbors, 
            minSize=min_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        fixed_confidence = 0.7
        if fixed_confidence < min_conf:
            return faces

        for (x, y, w, h) in face_rects:
            faces.append({
                'bbox': _normalize_bbox((x, y, x + w, y + h)),
                'confidence': fixed_confidence,  # Haar doesn't provide confidence, use fixed value
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


def analyze_face_attributes(bgr_image, face_bbox):
    """
    Analyze face for age, gender, emotion, and race using DeepFace
    """
    if not _HAS_DEEPFACE:
        return {}
    
    try:
        x1, y1, x2, y2 = face_bbox
        face_crop = bgr_image[y1:y2, x1:x2]
        
        if face_crop.size == 0:
            return {}
        
        # Convert to RGB for DeepFace
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        
        # Analyze using DeepFace
        analysis = DeepFace.analyze(
            face_rgb, 
            actions=['age', 'gender', 'emotion', 'race'],
            enforce_detection=False
        )
        
        if isinstance(analysis, list):
            analysis = analysis[0]
        
        return {
            'age': analysis.get('age', 'Unknown'),
            'gender': analysis.get('dominant_gender', 'Unknown'),
            'emotion': analysis.get('dominant_emotion', 'Unknown'),
            'race': analysis.get('dominant_race', 'Unknown'),
            'emotion_scores': analysis.get('emotion', {}),
            'gender_confidence': analysis.get('gender', {}).get(analysis.get('dominant_gender', ''), 0)
        }
    except Exception as e:
        print(f"Face analysis error: {e}")
        return {}


def detect_face_landmarks(bgr_image):
    """
    Detect facial landmarks using MediaPipe
    """
    landmarks_data = []
    if not _HAS_MEDIAPIPE:
        return landmarks_data, bgr_image
    
    try:
        output_image = bgr_image.copy()
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=10,
            refine_landmarks=True,
            min_detection_confidence=0.5
        ) as face_mesh:
            results = face_mesh.process(rgb)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw landmarks
                mp_drawing.draw_landmarks(
                    image=output_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                
                # Extract key landmark points
                h, w, _ = bgr_image.shape
                landmarks = []
                for landmark in face_landmarks.landmark:
                    landmarks.append({
                        'x': int(landmark.x * w),
                        'y': int(landmark.y * h),
                        'z': landmark.z
                    })
                landmarks_data.append(landmarks)
        
        return landmarks_data, output_image
    except Exception as e:
        print(f"Landmark detection error: {e}")
        return landmarks_data, bgr_image


def compare_faces(image1_path, image2_path):
    """
    Compare two faces and return similarity score
    """
    if not _HAS_FACE_RECOGNITION:
        return {'similarity': 0, 'are_same': False, 'error': 'face_recognition not available'}
    
    try:
        # Load and encode faces
        image1 = face_recognition.load_image_file(image1_path)
        image2 = face_recognition.load_image_file(image2_path)
        
        encodings1 = face_recognition.face_encodings(image1)
        encodings2 = face_recognition.face_encodings(image2)
        
        if not encodings1 or not encodings2:
            return {'similarity': 0, 'are_same': False, 'error': 'No faces found in one or both images'}
        
        # Compare faces
        face_distance = face_recognition.face_distance(encodings1, encodings2[0])[0]
        similarity = 1 - face_distance
        are_same = similarity > 0.6  # Threshold for same person
        
        return {
            'similarity': float(similarity),
            'are_same': bool(are_same),
            'distance': float(face_distance)
        }
    except Exception as e:
        return {'similarity': 0, 'are_same': False, 'error': str(e)}


def extract_face_crops(bgr_image, faces):
    """
    Extract individual face crops from the image
    """
    face_crops = []
    try:
        for i, face in enumerate(faces):
            x1, y1, x2, y2 = face['bbox']
            
            # Add padding
            padding = 20
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(bgr_image.shape[1], x2 + padding)
            y2 = min(bgr_image.shape[0], y2 + padding)
            
            face_crop = bgr_image[y1:y2, x1:x2]
            
            if face_crop.size > 0:
                # Convert to base64 for web transmission
                _, buffer = cv2.imencode('.jpg', face_crop)
                crop_base64 = base64.b64encode(buffer).decode('utf-8')
                
                face_crops.append({
                    'id': i,
                    'bbox': [x1, y1, x2, y2],
                    'image': f"data:image/jpeg;base64,{crop_base64}",
                    'confidence': face['confidence'],
                    'method': face['method']
                })
    except Exception as e:
        print(f"Face crop extraction error: {e}")
    
    return face_crops


def _get_method_color(method):
    color_map = {
        'MediaPipe': (0, 255, 0),
        'MTCNN': (255, 0, 0),
        'face_recognition': (0, 165, 255),
        'Haar': (255, 255, 0)
    }
    return color_map.get(method, (0, 255, 0))


def _draw_detected_faces(bgr_image, faces):
    """
    Draw normalized detector output onto an image and return serializable face details.
    """
    output_image = bgr_image.copy()
    face_details = []

    for face in faces:
        bbox = face.get('bbox')
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue

        face_id = len(face_details) + 1
        x1, y1, x2, y2 = _normalize_bbox(bbox)
        confidence = float(_to_python_number(face.get('confidence', 0.0) or 0.0))
        method = face.get('method', 'Unknown')
        color = _get_method_color(method)

        cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)

        label = f"Face {face_id} ({confidence:.2f})"
        detail = method
        label_y = max(25, y1 - 10)
        cv2.putText(
            output_image,
            label,
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )
        cv2.putText(
            output_image,
            detail,
            (x1, label_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1
        )

        face_details.append(_normalize_face_info({
            'id': face_id,
            'bbox': [x1, y1, x2, y2],
            'confidence': confidence,
            'method': method
        }))

    return output_image, face_details


def _get_detector_specs():
    return [
        {
            'key': 'mediapipe',
            'label': 'MediaPipe',
            'description': 'Fast Google detector optimized for modern images.',
            'available': lambda: _HAS_MEDIAPIPE and _MP_FACE_DETECTOR is not None,
            'runner': lambda image, min_conf: detect_faces_mediapipe(image, min_conf)
        },
        {
            'key': 'mtcnn',
            'label': 'MTCNN',
            'description': 'CNN-based detector that performs well on harder faces.',
            'available': lambda: _HAS_MTCNN and _MTCNN_DETECTOR is not None,
            'runner': lambda image, min_conf: detect_faces_mtcnn(image, min_conf)
        },
        {
            'key': 'face_recognition',
            'label': 'face_recognition',
            'description': 'HOG-based detector useful for profiles and encoding workflows.',
            'available': lambda: _HAS_FACE_RECOGNITION,
            'runner': lambda image, min_conf: detect_faces_face_recognition(image, min_conf)
        },
        {
            'key': 'haar',
            'label': 'Haar Cascade',
            'description': 'Classic OpenCV fallback detector with wide compatibility.',
            'available': lambda: True,
            'runner': lambda image, min_conf: detect_faces_haar(image, min_conf)
        }
    ]


def get_detector_catalog():
    """
    Return the detector list with availability metadata for UI or API usage.
    """
    catalog = []
    for spec in _get_detector_specs():
        catalog.append({
            'key': spec['key'],
            'label': spec['label'],
            'description': spec['description'],
            'available': bool(spec['available']())
        })
    return catalog


def compare_detection_modes(bgr_image, min_conf=0.5, detector_keys=None):
    """
    Run multiple detectors independently on the same image for side-by-side comparison.
    """
    processed_image = preprocess_image(bgr_image.copy())
    specs = _get_detector_specs()
    known_keys = {spec['key'] for spec in specs}
    requested_keys = [str(key).strip().lower() for key in (detector_keys or []) if str(key).strip()]
    requested_set = set(requested_keys)

    detector_results = []

    for spec in specs:
        if requested_set and spec['key'] not in requested_set:
            continue

        available = bool(spec['available']())
        detector_result = {
            'key': spec['key'],
            'label': spec['label'],
            'description': spec['description'],
            'available': available,
            'face_count': 0,
            'faces': [],
            'analysis_report': {},
            'summary': {
                'avg_confidence': None,
                'top_confidence': None,
                'detected_methods': []
            }
        }

        if not available:
            detector_result['error'] = f"{spec['label']} is not available in this environment."
            detector_results.append(detector_result)
            continue

        try:
            detected_faces = spec['runner'](processed_image.copy(), min_conf)
            output_image, face_details = _draw_detected_faces(processed_image, detected_faces)
            analysis_report = generate_face_analysis_report(face_details)
            confidence_stats = analysis_report.get('confidence_stats') or {}

            detector_result.update({
                'face_count': len(face_details),
                'faces': face_details,
                'analysis_report': analysis_report,
                'summary': {
                    'avg_confidence': confidence_stats.get('avg'),
                    'top_confidence': confidence_stats.get('max'),
                    'detected_methods': sorted({
                        face.get('method') for face in face_details if face.get('method')
                    })
                },
                'output_image': output_image
            })
        except Exception as e:
            detector_result['error'] = str(e)

        detector_results.append(detector_result)

    for requested_key in requested_keys:
        if requested_key in known_keys:
            continue
        detector_results.append({
            'key': requested_key,
            'label': requested_key.replace('_', ' ').title(),
            'description': 'Unknown detector key.',
            'available': False,
            'face_count': 0,
            'faces': [],
            'analysis_report': {},
            'summary': {
                'avg_confidence': None,
                'top_confidence': None,
                'detected_methods': []
            },
            'error': 'Unknown detector requested.'
        })

    successful_results = [
        result for result in detector_results
        if result.get('available') and not result.get('error')
    ]
    face_counts = [result['face_count'] for result in successful_results]

    if not face_counts:
        agreement = 'Unavailable'
        best_detectors = []
        max_faces = 0
        min_faces = 0
        count_spread = 0
        average_faces = 0
    else:
        max_faces = max(face_counts)
        min_faces = min(face_counts)
        count_spread = max_faces - min_faces
        average_faces = round(sum(face_counts) / len(face_counts), 2)
        best_detectors = []
        if max_faces > 0:
            best_detectors = [
                result['label'] for result in successful_results
                if result['face_count'] == max_faces
            ]
        if len(face_counts) <= 1:
            agreement = 'Single detector'
        elif count_spread == 0:
            agreement = 'Strong'
        elif count_spread == 1:
            agreement = 'Moderate'
        else:
            agreement = 'Mixed'

    return {
        'detectors': detector_results,
        'summary': {
            'requested_detectors': len(detector_results),
            'available_detectors': sum(1 for result in detector_results if result.get('available')),
            'successful_detectors': len(successful_results),
            'average_faces': average_faces,
            'max_faces': max_faces,
            'min_faces': min_faces,
            'count_spread': count_spread,
            'best_detectors': best_detectors,
            'agreement': agreement
        }
    }


def detect_and_draw_faces_realtime(bgr_image, min_conf=0.5):
    """
    Lightweight webcam pipeline that favors responsiveness over exhaustive analysis.
    """
    realtime_image = resize_for_realtime(bgr_image)
    if _HAS_MEDIAPIPE and _MP_FACE_DETECTOR is not None:
        final_faces = detect_faces_mediapipe(realtime_image, min_conf)
        detector_name = 'MediaPipe'
        if not final_faces:
            final_faces = detect_faces_haar(
                realtime_image,
                min_conf=min_conf * 0.7,
                scale_factor=1.1,
                min_neighbors=6,
                min_size=(40, 40)
            )
            if final_faces:
                detector_name = 'Haar'
    else:
        final_faces = detect_faces_haar(
            realtime_image,
            min_conf=min_conf * 0.7,
            scale_factor=1.1,
            min_neighbors=6,
            min_size=(40, 40)
        )
        detector_name = 'Haar'

    output_image = realtime_image.copy()
    face_details = []

    for i, face in enumerate(final_faces):
        x1, y1, x2, y2 = face['bbox']
        confidence = face['confidence']
        method = face['method']
        color = (0, 220, 110) if method == 'MediaPipe' else (255, 210, 0)

        cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            output_image,
            f"Face {i + 1} ({confidence:.2f})",
            (x1, max(22, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2
        )

        face_details.append({
            'id': i + 1,
            'bbox': [x1, y1, x2, y2],
            'confidence': confidence,
            'method': method
        })

    face_details = [_normalize_face_info(face) for face in face_details]
    return output_image, len(face_details), face_details, detector_name


def detect_and_draw_faces(bgr_image, min_conf=0.5, analysis_mode='basic'):
    """
    Enhanced face detection with multiple analysis modes:
    - 'basic': Just detection and bounding boxes
    - 'advanced': Detection + age, gender, emotion analysis
    - 'landmarks': Detection + facial landmarks
    - 'full': All features combined
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
    fr_faces = detect_faces_face_recognition(processed_image, min_conf)
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
    face_details = []
    landmarks_image = None
    
    # Apply facial landmarks if requested
    if analysis_mode in ['landmarks', 'full']:
        landmarks_data, landmarks_image = detect_face_landmarks(bgr_image)
    
    for i, face in enumerate(final_faces):
        x1, y1, x2, y2 = face['bbox']
        confidence = face['confidence']
        method = face['method']
        
        # Use different colors for different methods
        color = _get_method_color(method)
        
        # Draw rectangle
        cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)
        
        # Basic face info
        face_info = {
            'id': i + 1,
            'bbox': [x1, y1, x2, y2],
            'confidence': confidence,
            'method': method
        }
        
        # Advanced analysis if requested
        if analysis_mode in ['advanced', 'full']:
            attributes = analyze_face_attributes(bgr_image, (x1, y1, x2, y2))
            face_info.update(attributes)
            
            # Enhanced label with analysis
            if attributes:
                age = attributes.get('age', 'Unknown')
                gender = attributes.get('gender', 'Unknown')
                emotion = attributes.get('emotion', 'Unknown')
                
                label = f"Face {i+1}: {gender}, {age}y, {emotion}"
                label_detail = f"{method}: {confidence:.2f}"
            else:
                label = f"Face {i+1} ({method}: {confidence:.2f})"
                label_detail = ""
        else:
            label = f"Face {i+1} ({method}: {confidence:.2f})"
            label_detail = ""
        
        # Draw labels
        label_y = max(25, y1 - 10)
        cv2.putText(
            output_image, 
            label,
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6,
            color, 
            2
        )
        
        if label_detail:
            cv2.putText(
                output_image, 
                label_detail,
                (x1, label_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.4,
                color, 
                1
            )
        
        face_details.append(_normalize_face_info(face_info))
    
    # Use landmarks image if available
    if landmarks_image is not None:
        output_image = landmarks_image
    
    return output_image, face_count, face_details


def process_batch_images(image_paths, min_conf=0.5, analysis_mode='basic', output_dir=None):
    """
    Process multiple images for face detection
    """
    results = []
    
    if output_dir is None:
        # Use current working directory if not specified
        output_dir = os.path.join(os.getcwd(), 'outputs')
        os.makedirs(output_dir, exist_ok=True)
    
    for i, image_path in enumerate(image_paths):
        try:
            image = cv2.imread(image_path)
            if image is None:
                results.append({
                    'id': i,
                    'path': image_path,
                    'error': 'Could not read image',
                    'face_count': 0,
                    'faces': []
                })
                continue
            
            output_image, face_count, face_details = detect_and_draw_faces(
                image, min_conf, analysis_mode
            )
            
            # Save processed image
            output_filename = f"batch_result_{i}_{uuid.uuid4().hex[:8]}.jpg"
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, output_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            results.append({
                'id': i,
                'path': image_path,
                'output_path': output_path,
                'output_url': f"/static/outputs/{output_filename}",
                'face_count': face_count,
                'faces': face_details
            })
            
        except Exception as e:
            results.append({
                'id': i,
                'path': image_path,
                'error': str(e),
                'face_count': 0,
                'faces': []
            })
    
    return results


def generate_face_analysis_report(face_details):
    """
    Generate a comprehensive analysis report
    """
    if not face_details:
        return {}

    def _build_histogram(values, bins, labels=None):
        total = len(values)
        histogram = []
        for idx, (start, end) in enumerate(bins):
            if idx == len(bins) - 1:
                count = sum(1 for v in values if v >= start and v <= end)
            else:
                count = sum(1 for v in values if v >= start and v < end)
            percent = round((count / total) * 100, 1) if total else 0.0
            label = labels[idx] if labels and idx < len(labels) else f"{start:.1f}-{end:.1f}"
            histogram.append({
                'label': label,
                'range': [start, end],
                'count': count,
                'percent': percent
            })
        return histogram
    
    # Statistics
    total_faces = len(face_details)
    
    # Age statistics
    ages = [face.get('age') for face in face_details if face.get('age') != 'Unknown' and face.get('age')]
    avg_age = sum(ages) / len(ages) if ages else 0
    
    # Gender distribution
    genders = [face.get('gender') for face in face_details if face.get('gender') != 'Unknown']
    gender_counts = {gender: genders.count(gender) for gender in set(genders)}
    
    # Emotion distribution  
    emotions = [face.get('emotion') for face in face_details if face.get('emotion') != 'Unknown']
    emotion_counts = {emotion: emotions.count(emotion) for emotion in set(emotions)}
    
    # Detection method statistics
    methods = [face.get('method') for face in face_details]
    method_counts = {method: methods.count(method) for method in set(methods)}

    # Confidence statistics
    confidences = [face.get('confidence') for face in face_details if face.get('confidence') is not None]
    confidence_stats = None
    confidence_histogram = []
    if confidences:
        conf_min = min(confidences)
        conf_max = max(confidences)
        conf_avg = sum(confidences) / len(confidences)
        confidence_stats = {
            'min': round(conf_min, 3),
            'max': round(conf_max, 3),
            'avg': round(conf_avg, 3)
        }
        conf_bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        confidence_histogram = _build_histogram(confidences, conf_bins)

    # Face size statistics (relative to the largest detected face)
    areas = []
    for face in face_details:
        bbox = face.get('bbox')
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            width = max(0, x2 - x1)
            height = max(0, y2 - y1)
            area = width * height
            if area > 0:
                areas.append(area)

    face_size_stats = None
    face_size_histogram = []
    if areas:
        min_area = min(areas)
        max_area = max(areas)
        avg_area = sum(areas) / len(areas)
        face_size_stats = {
            'min_area': round(min_area, 1),
            'max_area': round(max_area, 1),
            'avg_area': round(avg_area, 1)
        }
        max_area_safe = max_area if max_area > 0 else 1
        ratios = [area / max_area_safe for area in areas]
        size_bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        size_labels = [
            "Tiny (0.0-0.2)",
            "Small (0.2-0.4)",
            "Medium (0.4-0.6)",
            "Large (0.6-0.8)",
            "XL (0.8-1.0)"
        ]
        face_size_histogram = _build_histogram(ratios, size_bins, size_labels)

    # Method breakdown with percentages
    method_breakdown = []
    for method, count in sorted(method_counts.items(), key=lambda x: x[1], reverse=True):
        percent = round((count / total_faces) * 100, 1) if total_faces else 0.0
        method_breakdown.append({
            'label': method,
            'count': count,
            'percent': percent
        })
    
    return {
        'total_faces': total_faces,
        'average_age': round(avg_age, 1) if avg_age > 0 else None,
        'age_range': [min(ages), max(ages)] if ages else None,
        'gender_distribution': gender_counts,
        'emotion_distribution': emotion_counts,
        'detection_methods': method_counts,
        'method_breakdown': method_breakdown,
        'confidence_stats': confidence_stats,
        'confidence_histogram': confidence_histogram,
        'face_size_stats': face_size_stats,
        'face_size_histogram': face_size_histogram,
        'faces': face_details
    }
