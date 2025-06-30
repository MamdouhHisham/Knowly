from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import threading
import numpy as np
import cv2
from face_analyzer import FaceAnalyzer

app = FastAPI()

analyzer = None 

@app.post("/start_tracking/")
def start_tracking():
    global analyzer
    analyzer = FaceAnalyzer()
    analyzer.start_session()
    return {"message": "Tracking session started"}

@app.post("/process_frame/")
async def process_frame(file: UploadFile = File(...)):
    global analyzer
    if analyzer is None:
        return {"error": "Tracking session not started"}

   
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    result, _, _ = analyzer.analyze(frame)
    if result is None:
        return {"message": "No face detected in frame"}

    return result 

@app.post("/stop_tracking/")
def stop_tracking():
    global analyzer
    if analyzer is None:
        return {"error": "No active session to stop"}

    report = analyzer.generate_report()
    analyzer.save_report()
    analyzer = None 
    return JSONResponse(content=report)
