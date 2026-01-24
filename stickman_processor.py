import cv2
import numpy as np
import time
from stickman import draw_stickman
from stickman_camera import StickmanCamera
from smoother import LandmarkSmoother
from pose_detectors import (MediaPipeHandDetector, YoloBodyDetector)

class StickmanProcessor:
    """
    Centralized processor for stickman pose detection and drawing.
    Uses Hybrid Detector: YOLO (Body) + MediaPipe (Hands).
    """
    def __init__(self):
        self.camera = None # Persistent camera instance
        
        # Smoothing
        self.single_pose_smoother = LandmarkSmoother(alpha=0.5)
        self.multi_pose_smoothers = {} # index -> LandmarkSmoother
        
        # Detectors
        self.body_detector = YoloBodyDetector()
        self.hand_detector = MediaPipeHandDetector()

    def process_frame(self, frame, thickness=4, auto_track=False, single_mode=False, camera_instance=None):
        """
        Process a single frame (BGR or RGB numpy array).
        Returns the processed stickman image (BGR).
        """
        if frame is None: return None, camera_instance
        
        if camera_instance is None:
             if self.camera is None and auto_track:
                 self.camera = StickmanCamera()
             cam_to_use = self.camera if auto_track else None
        else:
             cam_to_use = camera_instance if auto_track else None

        # Resize for performance if needed (max 640w)
        h, w = frame.shape[:2]
        if w > 640:
            scale = 640 / w
            new_h = int(h * scale)
            frame = cv2.resize(frame, (640, new_h))
            
        try:
            # 1. Detect Body
            body_res = self.body_detector.process(frame, single_mode=single_mode)
            
            # 2. Detect Hands
            hand_res = self.hand_detector.process(frame)
            camera_hands = hand_res.hands
            
            pose_landmarks_list = body_res.poses
            
            # Apply Smoothing (Multi Mode)
            smoothed_poses = []
            if pose_landmarks_list:
                for i, landmarks in enumerate(pose_landmarks_list):
                    # JUMP DETECTION
                    if i in self.multi_pose_smoothers:
                        smoother = self.multi_pose_smoothers[i]
                        if len(landmarks) > 0 and 0 in smoother.smoothed_landmarks:
                            new_nose = landmarks[0]
                            prev_nose = smoother.smoothed_landmarks[0]
                            dist = np.sqrt((new_nose.x - prev_nose.x)**2 + (new_nose.y - prev_nose.y)**2)
                            if dist > 0.2:
                                self.multi_pose_smoothers[i] = LandmarkSmoother(alpha=0.5)

                    if i not in self.multi_pose_smoothers:
                        self.multi_pose_smoothers[i] = LandmarkSmoother(alpha=0.5)
                    
                    s_lms = self.multi_pose_smoothers[i].update(landmarks)
                    smoothed_poses.append(s_lms)
            
            output = draw_stickman(
                smoothed_poses, 
                img_shape=frame.shape, 
                thickness=int(thickness), 
                camera=cam_to_use, 
                mode='multi',
                multi_hand_landmarks=camera_hands
            )
            
            return output, cam_to_use

        except Exception as e:
            print(f"Error in processor: {e}")
            import traceback
            traceback.print_exc()
            return np.ones((h, w, 3), dtype=np.uint8) * 255, cam_to_use
