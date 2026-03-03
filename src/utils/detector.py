import cv2

try:
    import mediapipe as mp
except Exception:
    mp = None

_HAS_MEDIAPIPE = bool(mp) and hasattr(mp, "solutions") and hasattr(mp.solutions, "face_detection")

if _HAS_MEDIAPIPE:
    mp_face = mp.solutions.face_detection
else:
    mp_face = None
    _HAAR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    _HAAR = cv2.CascadeClassifier(_HAAR_PATH)
    if _HAAR.empty():
        raise RuntimeError(f"Could not load Haar cascade from {_HAAR_PATH}")

def detect_and_draw_faces(bgr_image, min_conf=0.5):
    face_count = 0
    h, w, _ = bgr_image.shape

    if _HAS_MEDIAPIPE:
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        with mp_face.FaceDetection(model_selection=0, min_detection_confidence=min_conf) as face_detection:
            results = face_detection.process(rgb)

        if results.detections:
            for det in results.detections:
                face_count += 1
                bbox = det.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                bw = int(bbox.width * w)
                bh = int(bbox.height * h)

                x = max(0, x)
                y = max(0, y)
                x2 = min(w - 1, x + bw)
                y2 = min(h - 1, y + bh)

                cv2.rectangle(bgr_image, (x, y), (x2, y2), (0, 255, 0), 2)
                score = det.score[0]
                cv2.putText(
                    bgr_image, f"Face {score:.2f}",
                    (x, max(20, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2
                )
    else:
        # Fallback for environments where MediaPipe is unavailable (e.g., Python 3.13)
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        faces = _HAAR.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, bw, bh) in faces:
            face_count += 1
            x2 = min(w - 1, x + bw)
            y2 = min(h - 1, y + bh)
            cv2.rectangle(bgr_image, (x, y), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                bgr_image, "Face",
                (x, max(20, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0), 2
            )

    return bgr_image, face_count
