import cv2
import dlib


class FaceDetector:
    """Unified face detection using dlib."""
    def __init__(self, model_path="trained_models/shape_predictor_68_face_landmarks.dat"):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(model_path)

    def detect_faces(self, frame):
        """Detect faces and return bounding boxes and landmarks."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        if not faces:
            return []

        results = []
        for face in faces:
            landmarks = self.predictor(gray, face)
            x, y = face.left(), face.top()
            w, h = face.right() - x, face.bottom() - y
            results.append({
                "bbox": (x, y, w, h),
                "landmarks": landmarks,
                "nose_tip": (landmarks.part(30).x, landmarks.part(30).y)
            })
        return results