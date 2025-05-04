import cv2
import numpy as np
from transformers import ViTForImageClassification, ViTImageProcessor #,ViTFeatureExtractor
import torch
from collections import Counter
class EmotionDetector:
    def __init__(self):
        self.model_name = "trpakov/vit-face-expression"
        try:
            self.processor = ViTImageProcessor.from_pretrained(self.model_name)
            self.model = ViTForImageClassification.from_pretrained(self.model_name)
        except Exception as e:
            raise RuntimeError(f"Model loading error: {e}")
        
        self.emotion_labels = {
            0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy",
            4: "Neutral", 5: "Sad", 6: "Surprise"
        }
        self.emotion_list = []  # Track emotions for summary

    def detect_emotion(self, face_img):
        """Detect emotion from a face image."""
        if face_img is None or face_img.size == 0:
            print("face_img is empty or None")
            return None

        try:
            # Convert BGR to RGB
            rgb_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            # Process image for ViT model
            inputs = self.processor(images=rgb_img, return_tensors="pt")
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                predicted_class = torch.argmax(logits, dim=1).item()
            
            # Get emotion label
            emotion = self.emotion_labels.get(predicted_class, "Unknown")
            self.emotion_list.append(emotion)  # Store for summary
            
            return emotion
        
        except Exception as e:
            print(f"Analysis error: {e}")
            return None

    def get_emotion_summary(self):
        """Return a summary of detected emotions."""
        if not self.emotion_list:
            return {"message": "No emotions detected"}
        
        emotion_counter = Counter(self.emotion_list)
        most_common_emotion = emotion_counter.most_common(1)[0][0]
        return {
            "most_common_emotion": most_common_emotion
        }