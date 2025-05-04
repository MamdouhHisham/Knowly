import cv2
import json
import os
from datetime import datetime
from core.face_detector import FaceDetector
from core.utils import preprocess_frame
from modules.head_pose.orientation import HeadOrientation
from modules.eye_tracking.gaze_tracker import GazeTracker
from modules.emotion.emotion_detector import EmotionDetector

class FaceAnalyzer:
    def __init__(self):
        self.face_detector = FaceDetector()
        self.head_orientation = HeadOrientation()
        self.gaze_tracker = GazeTracker()
        self.emotion_detector = EmotionDetector()
        self.session_start = None
        self.session_end = None
        self.reports_dir = "session_reports"
        os.makedirs(self.reports_dir, exist_ok=True)
        self.focus_frames = 0
        self.total_frames = 0
        self.tracking_quality = 1.0  # Default to perfect tracking
        self.valid_frames = 0  # Frames where we have either head or eye tracking

    def start_session(self):
        """Initialize a new tracking session"""
        self.session_start = datetime.now()
        self.session_end = None
        self.focus_frames = 0
        self.total_frames = 0
        self.valid_frames = 0

    def analyze(self, frame):
        """Analyze frame for head pose, gaze, and emotion"""
        processed_frame = preprocess_frame(frame)
        faces = self.face_detector.detect_faces(processed_frame)
        
        self.total_frames += 1
        
        if not faces:
            return None, processed_frame, None

        face_data = faces[0]
        bbox, landmarks, nose_tip = face_data["bbox"], face_data["landmarks"], face_data["nose_tip"]
        x, y, w, h = bbox

        # Head pose estimation
        head_pose = self.head_orientation.estimate_pose(processed_frame, bbox)
        if head_pose:
            processed_frame = self.head_orientation.draw_axis(
                processed_frame, head_pose["yaw"], head_pose["pitch"], 
                head_pose["roll"], nose_tip[0], nose_tip[1], size=bbox[2]//2
            )
            cv2.rectangle(processed_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Gaze tracking
        gaze_dir = self.gaze_tracker.analyze(processed_frame, landmarks)
        annotated_frame = self.gaze_tracker.annotated_frame()

        # Emotion detection
        face_img = processed_frame[y:y+h, x:x+w]
        emotion = self.emotion_detector.detect_emotion(face_img) if face_img.size > 0 else None
        
        # Check if we have valid tracking data (either head pose or gaze)
        has_valid_tracking = (head_pose is not None) or (self.gaze_tracker.pupils_located)
        if has_valid_tracking:
            self.valid_frames += 1
        
        # Determine if user is focused (looking forward and center)
        forward_center = False
        if head_pose and gaze_dir:
            # Full focus - both systems working and indicating attention
            if head_pose['orientation'] == 'forward' and gaze_dir == 'center':
                forward_center = True
                self.focus_frames += 1.0
            # Partial focus - head forward but eyes not center
            elif head_pose['orientation'] == 'forward':
                forward_center = True  # Still consider this focused
                self.focus_frames += 0.7
            # Partial focus - eyes center but head not forward
            elif gaze_dir == 'center':
                forward_center = True  # Still consider this focused
                self.focus_frames += 0.7
    
        # If we only have head pose data
        elif head_pose and head_pose['orientation'] == 'forward':
            forward_center = True
            self.focus_frames += 0.7
    
        # If we only have gaze data
        elif gaze_dir == 'center':
            forward_center = True
            self.focus_frames += 0.7

    
        return {
            "head_pose": head_pose,
            "gaze": {
                "horizontal": self.gaze_tracker.horizontal_ratio() if self.gaze_tracker.pupils_located else None,
                "vertical": self.gaze_tracker.vertical_ratio() if self.gaze_tracker.pupils_located else None,
                "is_left": self.gaze_tracker.is_left(),
                "is_right": self.gaze_tracker.is_right(),
                "is_center": self.gaze_tracker.is_center(),
                "is_blinking": self.gaze_tracker.is_blinking()
            },
            "emotion": emotion
        }, annotated_frame, forward_center

    def set_tracking_quality(self, quality):
        """Store the tracking quality for reporting"""
        self.tracking_quality = quality

    def get_summaries(self):
        """Return all summaries in one dictionary"""
        return {
            "gaze_tracker": self.gaze_tracker.get_gaze_summary(),
            "head_pose": self.head_orientation.get_pose_summary(),
            "emotion": self.emotion_detector.get_emotion_summary(),
        }

    def calculate_focus_percentage(self):
        """Calculate focus percentage with compensation for tracking quality"""
        if self.total_frames == 0:
            return 0
            
        if self.valid_frames == 0:
            return 0
        # Calculate raw focus percentage
        raw_focus = (self.focus_frames / self.valid_frames) * 100
        
        # Adjust focus calculation based on tracking quality
        if self.tracking_quality < 0.5:
            # If tracking quality is poor, we need to be more conservative
            # This formula increases uncertainty as tracking quality decreases
            adjusted_focus = raw_focus * (0.5 + self.tracking_quality/2)
        else:
            adjusted_focus = raw_focus
            
        return adjusted_focus

    def generate_report(self):
        """Generate session report with timing and summaries"""
        if not self.session_start:
            return None

        self.session_end = datetime.now()
        
        # Calculate tracking quality
        if not hasattr(self, 'tracking_quality'):
            self.tracking_quality = self.valid_frames / self.total_frames if self.total_frames > 0 else 0

        return {
            "session_info": {
                "start_time": self.session_start.isoformat(),
                "end_time": self.session_end.isoformat(),
                "duration_seconds": (self.session_end - self.session_start).total_seconds()
            },
            "analysis_summary": self.get_summaries(),
            "focus_analysis": {
                "focused_frames": self.focus_frames,
                "total_frames": self.total_frames,
                "valid_frames": self.valid_frames,
                "tracking_quality": self.tracking_quality,
                "focus_percentage": self.calculate_focus_percentage()
            }
        }

    def save_report(self, filename=None):
        """Save the report to JSON file"""
        report = self.generate_report()
        if not report:
            return False

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"session_report_{timestamp}.json"
            
        filepath = os.path.join(self.reports_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        return True