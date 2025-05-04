import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import math
from .model import Hopenet, Bottleneck
from collections import Counter


class HeadOrientation:
    def __init__(self, model_path="trained_models/hopenet_robust_alpha1.pkl"):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = Hopenet(block=Bottleneck, layers=[3, 4, 6, 3], num_bins=66).to(self.device)
        self.pose_list=[]
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device,weights_only=True))
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.idx_tensor = torch.FloatTensor(list(range(66))).to(self.device)

    def estimate_pose(self, frame, bbox):
        """Estimate head pose from face region."""
        x, y, w, h = bbox
        face_img = frame[y:y+h, x:x+w]
        if face_img.size == 0:
            return None

        try:
            pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
            img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                yaw, pitch, roll = self.model(img_tensor)
                yaw_pred = torch.sum(torch.softmax(yaw, dim=1) * self.idx_tensor, dim=1) * 3 - 99
                pitch_pred = torch.sum(torch.softmax(pitch, dim=1) * self.idx_tensor, dim=1) * 3 - 99
                roll_pred = torch.sum(torch.softmax(roll, dim=1) * self.idx_tensor, dim=1) * 3 - 99

            yaw_value, pitch_value, roll_value = yaw_pred.item(), pitch_pred.item(), roll_pred.item()
            orientation = self._get_head_orientation(yaw_value, pitch_value)
            self.pose_list.append(orientation)
            return {"yaw": yaw_value, "pitch": pitch_value, "roll": roll_value, "orientation": orientation}
        except Exception:
            return None

    def draw_axis(self, img, yaw, pitch, roll, tdx, tdy, size=50):
        pitch = pitch * np.pi / 180
        yaw = -(yaw * np.pi / 180)
        roll = roll * np.pi / 180
        x1 = size * (math.cos(yaw) * math.cos(roll)) + tdx
        y1 = size * (math.cos(pitch) * math.sin(roll) + math.cos(roll) * math.sin(pitch) * math.sin(yaw)) + tdy
        x2 = size * (-math.cos(yaw) * math.sin(roll)) + tdx
        y2 = size * (math.cos(pitch) * math.cos(roll) - math.sin(pitch) * math.sin(yaw) * math.sin(roll)) + tdy
        x3 = size * (math.sin(yaw)) + tdx
        y3 = size * (-math.cos(yaw) * math.sin(pitch)) + tdy
        cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 3)
        cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 3)
        cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 3)
        return img

    @staticmethod
    def _get_head_orientation(yaw, pitch):
        yaw_threshold, pitch_threshold = 7, 7
        if abs(yaw) < yaw_threshold and abs(pitch) < pitch_threshold:
            return "forward"
        elif yaw > yaw_threshold:
            return "right"
        elif yaw < -yaw_threshold:
            return "left"
        elif pitch > pitch_threshold:
            return "up"
        elif pitch < -pitch_threshold:
            return "down"
        return "forward"
    

    def get_pose_summary(self):
        """Return a summary of detected pose."""
        if not self.pose_list:
            return {"message": "No pose detected"}
        
        pose_counter = Counter(self.pose_list)
        most_common_pose = pose_counter.most_common(1)[0][0]
        return {
            "most_common_head_pose": most_common_pose
        }