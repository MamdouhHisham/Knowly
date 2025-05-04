from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse
import threading
import cv2
import time
import os

from face_analyzer import FaceAnalyzer
from core.utils import get_video_feed

app = FastAPI()

analyzer = None
cap = None
tracking_thread = None
running = False

def tracking_loop():
    global analyzer, cap, running
    while running:
        ret, frame = cap.read()
        if not ret:
            break
        analyzer.analyze(frame)
        time.sleep(0.03)  # ~30 FPS

@app.post("/start_tracking/")
def start_tracking():
    global analyzer, cap, tracking_thread, running
    if running:
        return {"message": "Tracking is already running."}

    analyzer = FaceAnalyzer()
    cap = get_video_feed()
    if not cap.isOpened():
        return {"error": "Camera could not be opened."}

    analyzer.start_session()
    running = True
    tracking_thread = threading.Thread(target=tracking_loop)
    tracking_thread.start()

    return {"message": "Tracking started successfully."}

@app.post("/stop_tracking/")
def stop_tracking():
    global analyzer, cap, tracking_thread, running

    if not running:
        return {"error": "No tracking session is active."}

    running = False
    tracking_thread.join()
    cap.release()
    report = analyzer.generate_report()
    analyzer.save_report()

    return JSONResponse(content=report)

