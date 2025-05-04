import cv2
from .config import SOURCE, VIDEO_PATH, CAMERA_ID


def get_video_feed():
    """Initialize and return video capture object based on config"""
    if SOURCE == "realtime":
        cap = cv2.VideoCapture(CAMERA_ID)
    elif SOURCE == "video":
        cap = cv2.VideoCapture(VIDEO_PATH)
    return cap


def preprocess_frame(frame, width=640, height=480):
    """Preprocess frame"""
    return cv2.resize(frame, (width, height))