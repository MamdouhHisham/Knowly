import cv2
from .eye import Eye
from .calibration import Calibration
from collections import Counter

class GazeTracker:
    def __init__(self):
        self.frame = None
        self.eye_left = None
        self.eye_right = None
        self.calibration = Calibration()
        self.gaze_list=[]

    @property
    def pupils_located(self):
        try:
            return all(isinstance(coord, int) for coord in [
                self.eye_left.pupil.x, self.eye_left.pupil.y,
                self.eye_right.pupil.x, self.eye_right.pupil.y
            ])
        except Exception:
            return False

    
    def analyze(self, frame, landmarks):
        self.frame = frame
        gaze_dir=""
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        try:
            self.eye_left = Eye(gray_frame, landmarks, 0, self.calibration)
            self.eye_right = Eye(gray_frame, landmarks, 1, self.calibration)
            
            # Track gaze direction when pupils are located
            if self.pupils_located:
                if self.is_left():
                    self.gaze_list.append("left")
                    gaze_dir='left'
                elif self.is_right():
                    self.gaze_list.append("right")
                    gaze_dir='right'
                else:
                    self.gaze_list.append("center")
                    gaze_dir='center'

                # Track blinking
                if self.is_blinking():
                    self.gaze_list.append("blink")
                    
        except Exception:
            self.eye_left = self.eye_right = None
        
        return gaze_dir
        
 


    def get_gaze_summary(self):
        """Return a summary of detected gaze directions."""
        if not self.gaze_list:
            return {"message": "No gaze data detected"}
        
        gaze_counter = Counter(self.gaze_list)
        most_common_gaze = gaze_counter.most_common(1)[0][0]

        return {
            "most_common_gaze": most_common_gaze,
            "blink_count": gaze_counter.get("blink", 0)
        }

    def pupil_left_coords(self):
        if self.pupils_located:
            return (self.eye_left.origin[0] + self.eye_left.pupil.x,
                    self.eye_left.origin[1] + self.eye_left.pupil.y)

    def pupil_right_coords(self):
        if self.pupils_located:
            return (self.eye_right.origin[0] + self.eye_right.pupil.x,
                    self.eye_right.origin[1] + self.eye_right.pupil.y)

    def horizontal_ratio(self):
        if self.pupils_located:
            pupil_left = self.eye_left.pupil.x / (self.eye_left.center[0] * 2 - 10)
            pupil_right = self.eye_right.pupil.x / (self.eye_right.center[0] * 2 - 10)
            return (pupil_left + pupil_right) / 2

    def vertical_ratio(self):
        if self.pupils_located:
            pupil_left = self.eye_left.pupil.y / (self.eye_left.center[1] * 2 - 10)
            pupil_right = self.eye_right.pupil.y / (self.eye_right.center[1] * 2 - 10)
            return (pupil_left + pupil_right) / 2

    def is_right(self):
        return self.pupils_located and self.horizontal_ratio() <= 0.50

    def is_left(self):
        return self.pupils_located and self.horizontal_ratio() >= 0.80

    def is_center(self):
        return self.pupils_located and not (self.is_right() or self.is_left())

    def is_blinking(self):
        if self.pupils_located:
            blinking_ratio = (self.eye_left.blinking + self.eye_right.blinking) / 2
            return blinking_ratio > 3.8

    def annotated_frame(self):
        frame = self.frame.copy()
        if self.pupils_located:
            color = (0, 255, 0)
            x_left, y_left = self.pupil_left_coords()
            x_right, y_right = self.pupil_right_coords()
            cv2.line(frame, (x_left - 5, y_left), (x_left + 5, y_left), color)
            cv2.line(frame, (x_left, y_left - 5), (x_left, y_left + 5), color)
            cv2.line(frame, (x_right - 5, y_right), (x_right + 5, y_right), color)
            cv2.line(frame, (x_right, y_right - 5), (x_right, y_right + 5), color)
        return frame