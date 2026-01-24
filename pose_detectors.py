import cv2
import numpy as np
import time
import os
import urllib.request
from typing import List, Any, Optional

# Protocol for Landmarks
class NormalizedLandmark:
    def __init__(self, x, y, visibility=1.0, z=0.0):
        self.x = x
        self.y = y
        self.visibility = visibility
        self.z = z

class BodyResult:
    def __init__(self, poses: List[List[NormalizedLandmark]], raw: Any = None):
        self.poses = poses
        self.raw = raw

class HandResult:
    def __init__(self, hands: List[List[NormalizedLandmark]], raw: Any = None):
        self.hands = hands
        self.raw = raw

class BodyDetector:
    def process(self, frame: np.ndarray, single_mode: bool = False) -> BodyResult:
        raise NotImplementedError

class HandDetector:
    def process(self, frame: np.ndarray) -> HandResult:
        raise NotImplementedError

class MediaPipeHandDetector(HandDetector):
    def __init__(self):
        import mediapipe as mp
        self.mp = mp
        self.mp_hand = mp.tasks.vision.HandLandmarker
        self.hand_landmarker = None
        self.hand_model_path = 'hand_landmarker.task'

    def _ensure_model(self, path, url):
        if not os.path.exists(path):
            print(f"Model {path} not found. Downloading...")
            try:
                urllib.request.urlretrieve(url, path)
            except Exception as e:
                print(f"Failed to download {path}: {e}")

    def _init_model(self):
        if self.hand_landmarker: return
        self._ensure_model(self.hand_model_path, 
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
            
        hand_options = self.mp.tasks.vision.HandLandmarkerOptions(
            base_options=self.mp.tasks.BaseOptions(model_asset_path=self.hand_model_path),
            running_mode=self.mp.tasks.vision.RunningMode.VIDEO,
            num_hands=10,
            min_hand_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        )
        self.hand_landmarker = self.mp_hand.create_from_options(hand_options)

    def process(self, frame: np.ndarray) -> HandResult:
        self._init_model()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame.shape[2] == 3 else frame
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = int(time.time() * 1000)
        
        hand_res = self.hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        
        hands = []
        if hand_res.hand_landmarks:
            hands = hand_res.hand_landmarks
            
        return HandResult(hands=hands)


class YoloBodyDetector(BodyDetector):
    def __init__(self):
        from ultralytics import YOLO
        print("Loading YOLOv8-pose model...")
        self.model = YOLO('yolov8n-pose.pt')
        print("YOLO model loaded.")
        
        # Mapping COCO keypoints (17) to MediaPipe BlazePose (33)
        self.map_coco_to_mp = {
             0: 0,  # Nose
             1: 2,  # L Eye
             2: 5,  # R Eye
             3: 7,  # L Ear
             4: 8,  # R Ear
             5: 11, # L Shoulder
             6: 12, # R Shoulder
             7: 13, # L Elbow
             8: 14, # R Elbow
             9: 15, # L Wrist
             10: 16,# R Wrist
             11: 23,# L Hip
             12: 24,# R Hip
             13: 25,# L Knee
             14: 26,# R Knee
             15: 27,# L Ankle
             16: 28 # R Ankle
        }

    def process(self, frame: np.ndarray, single_mode: bool = False) -> BodyResult:
        # Run inference
        results = self.model(frame, verbose=False, stream=False)
        
        poses = []
        
        for r in results:
            if r.keypoints is not None:
                xyn = r.keypoints.xyn.cpu().numpy() # Normalized x, y
                d = r.keypoints.conf # Confidence
                if d is None:
                    d = np.ones((xyn.shape[0], xyn.shape[1]))
                else:
                    d = d.cpu().numpy()
                
                num_persons = xyn.shape[0]
                
                for i in range(num_persons):
                    mp_landmarks = [NormalizedLandmark(0, 0, visibility=0.0) for _ in range(33)]
                    
                    coco_kpts = xyn[i]
                    coco_confs = d[i]
                    
                    for coco_idx, mp_idx in self.map_coco_to_mp.items():
                        x, y = coco_kpts[coco_idx]
                        conf = coco_confs[coco_idx]
                        
                        if conf > 0.3 and not (x == 0 and y == 0):
                            mp_landmarks[mp_idx] = NormalizedLandmark(x, y, visibility=float(conf))
                            
                    poses.append(mp_landmarks)
                    
                    if single_mode: break 

        return BodyResult(poses=poses)
