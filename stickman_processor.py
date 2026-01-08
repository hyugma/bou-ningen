import cv2
import mediapipe as mp
import numpy as np
import time
import os
import urllib.request
from stickman import draw_stickman, StickmanCamera

class StickmanProcessor:
    """
    Centralized processor for stickman pose detection and drawing.
    Handles Single (Holistic) and Multi (Pose+Hand) modes.
    """
    def __init__(self):
        self.mp_pose = mp.tasks.vision.PoseLandmarker
        self.mp_hand = mp.tasks.vision.HandLandmarker
        self.mp_holistic = mp.solutions.holistic
        
        self.pose_landmarker = None
        self.hand_landmarker = None
        self.holistic = None
        self.camera = None # Persistent camera instance
        
        # Paths
        self.pose_model_path = 'pose_landmarker_full.task'
        self.hand_model_path = 'hand_landmarker.task'

    def _ensure_model(self, path, url):
        if not os.path.exists(path):
            print(f"Model {path} not found. Downloading...")
            try:
                urllib.request.urlretrieve(url, path)
                print(f"Downloaded {path}")
            except Exception as e:
                print(f"Failed to download {path}: {e}")

    def _init_multi_models(self):
        if self.pose_landmarker and self.hand_landmarker: return

        # Pose Model
        self._ensure_model(self.pose_model_path, 
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task")
        
        base_options = mp.tasks.BaseOptions(model_asset_path=self.pose_model_path)
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=5,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False
        )
        self.pose_landmarker = self.mp_pose.create_from_options(options)

        # Hand Model
        self._ensure_model(self.hand_model_path, 
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
            
        hand_options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self.hand_model_path),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=10,
            min_hand_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        )
        self.hand_landmarker = self.mp_hand.create_from_options(hand_options)

    def _init_single_model(self):
        if self.holistic: return
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        )

    def process_frame(self, frame, thickness=4, sketch_mode=False, auto_zoom=False, single_mode=False, camera_instance=None):
        """
        Process a single frame (BGR or RGB numpy array).
        Returns the processed stickman image (BGR).
        """
        if frame is None: return None, camera_instance
        
        # Handle camera instance logic
        # If caller passes a state (like Gradio), use it. 
        # But if we maintain internal state (Native), use self.camera.
        # Let's standardize: pass camera_instance if stateless (Gradio), else internal.
        if camera_instance is None:
             if self.camera is None and auto_zoom:
                 self.camera = StickmanCamera()
             cam_to_use = self.camera if auto_zoom else None
        else:
             cam_to_use = camera_instance if auto_zoom else None
             # If passed instance creates one, we should probably update it? 
             # Gradio State logic handles this by returning it.

        # Resize for performance if needed (max 640w) - Optional, but good practice
        h, w = frame.shape[:2]
        if w > 640:
            scale = 640 / w
            new_h = int(h * scale)
            frame = cv2.resize(frame, (640, new_h))
            
        try:
            if single_mode:
                self._init_single_model()
                # Holistic needs RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame.shape[2] == 3 else frame
                # Make non-writeable for MP
                frame_rgb.flags.writeable = False
                results = self.holistic.process(frame_rgb)
                
                output = draw_stickman(
                    results, 
                    img_shape=frame.shape, 
                    thickness=int(thickness), 
                    sketch_mode=sketch_mode, 
                    camera=cam_to_use, 
                    mode='single'
                )

            else:
                self._init_multi_models()
                # Tasks API needs RGB mp.Image
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame.shape[2] == 3 else frame
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                timestamp_ms = int(time.time() * 1000)
                
                pose_res = self.pose_landmarker.detect_for_video(mp_image, timestamp_ms)
                hand_res = self.hand_landmarker.detect_for_video(mp_image, timestamp_ms)
                
                output = draw_stickman(
                    pose_res.pose_landmarks, 
                    img_shape=frame.shape, 
                    thickness=int(thickness), 
                    sketch_mode=sketch_mode, 
                    camera=cam_to_use, 
                    mode='multi',
                    multi_hand_landmarks=hand_res.hand_landmarks
                )
            
            # If internal camera was used and not returned, it persists in self.camera
            return output, cam_to_use

        except Exception as e:
            print(f"Error in processor: {e}")
            return np.ones((h, w, 3), dtype=np.uint8) * 255, cam_to_use
