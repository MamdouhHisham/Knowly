# Configuration settings
MODEL_PATHS = {
    "landmarks": "trained_models/shape_predictor_68_face_landmarks.dat",
    "hopenet": "trained_models/hopenet_robust_alpha1.pkl"
}

# Input source 
SOURCE = "realtime"  # "realtime" or "video"
VIDEO_PATH = "C:\\Users\\MH\\Downloads\\Telegram Desktop\\video_2025-03-09_23-43-24.mp4"  # video file path if SOURCE is "video"
CAMERA_ID = 0  # Camera index for realtime input