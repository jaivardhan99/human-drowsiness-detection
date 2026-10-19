"""Face landmark extraction and EAR/MAR geometry using MediaPipe Face Mesh."""
import cv2
import numpy as np
import mediapipe as mp

# MediaPipe Face Mesh landmark indices (468-point model)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [61, 291, 39, 181, 0, 17, 269, 405]

LEFT_EYE_BOX = [362, 385, 387, 263, 373, 380, 374, 386]
RIGHT_EYE_BOX = [33, 160, 158, 133, 153, 144, 145, 163]


def _dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


def eye_aspect_ratio(points):
    p1, p2, p3, p4, p5, p6 = points
    vertical = _dist(p2, p6) + _dist(p3, p5)
    horizontal = 2.0 * _dist(p1, p4)
    return vertical / horizontal if horizontal > 0 else 0.0


def mouth_aspect_ratio(points):
    p1, p2, p3, p4, p5, p6, p7, p8 = points
    vertical = _dist(p3, p8) + _dist(p4, p7) + _dist(p5, p6)
    horizontal = 3.0 * _dist(p1, p2)
    return vertical / horizontal if horizontal > 0 else 0.0


class FaceLandmarker:
    """Wraps MediaPipe Face Mesh to extract EAR, MAR, and eye-crop images per frame."""

    def __init__(self, eye_crop_size=48):
        self.eye_crop_size = eye_crop_size
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def _pixel_points(self, landmarks, indices, w, h):
        return [(landmarks[i].x * w, landmarks[i].y * h) for i in indices]

    def _crop_eye(self, frame, landmarks, indices, w, h):
        pts = np.array(self._pixel_points(landmarks, indices, w, h), dtype=np.int32)
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        pad_x = int((x1 - x0) * 0.3) + 2
        pad_y = int((y1 - y0) * 0.5) + 2
        x0, y0 = max(0, x0 - pad_x), max(0, y0 - pad_y)
        x1, y1 = min(w, x1 + pad_x), min(h, y1 + pad_y)
        if x1 <= x0 or y1 <= y0:
            return np.zeros((self.eye_crop_size, self.eye_crop_size), dtype=np.uint8)
        crop = frame[y0:y1, x0:x1]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, (self.eye_crop_size, self.eye_crop_size))

    def process(self, frame_bgr):
        """Returns dict with ear, mar, left_eye_crop, right_eye_crop, or None if no face found."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.mesh.process(rgb)
        if not result.multi_face_landmarks:
            return None

        landmarks = result.multi_face_landmarks[0].landmark

        left_pts = self._pixel_points(landmarks, LEFT_EYE, w, h)
        right_pts = self._pixel_points(landmarks, RIGHT_EYE, w, h)
        mouth_pts = self._pixel_points(landmarks, MOUTH, w, h)

        ear = (eye_aspect_ratio(left_pts) + eye_aspect_ratio(right_pts)) / 2.0
        mar = mouth_aspect_ratio(mouth_pts)

        left_crop = self._crop_eye(frame_bgr, landmarks, LEFT_EYE_BOX, w, h)
        right_crop = self._crop_eye(frame_bgr, landmarks, RIGHT_EYE_BOX, w, h)

        return {
            "ear": ear,
            "mar": mar,
            "left_eye_crop": left_crop,
            "right_eye_crop": right_crop,
        }

    def close(self):
        self.mesh.close()
