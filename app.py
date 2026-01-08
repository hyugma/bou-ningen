import gradio as gr
import cv2
import numpy as np
import mediapipe as mp
from stickman import draw_stickman, StickmanCamera

# Initialize MediaPipe Pose Landmarker
model_path = 'pose_landmarker_full.task'
import os
import time
import urllib.request

if not os.path.exists(model_path):
    print(f"Model file {model_path} not found. Downloading...")
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
    try:
        urllib.request.urlretrieve(url, model_path)
        print("Download successful.")
    except Exception as e:
        print(f"Failed to download model: {e}")

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO, 
    num_poses=5,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_segmentation_masks=False
)

# Global landmarker instance
landmarker = PoseLandmarker.create_from_options(options)

# Global Hand Landmarker
hand_model_path = 'hand_landmarker.task'
import os
import urllib.request
if not os.path.exists(hand_model_path):
    print(f"Model file {hand_model_path} not found. Downloading...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    try:
        urllib.request.urlretrieve(url, hand_model_path)
    except Exception as e:
        print(f"Failed to download hand model: {e}")

HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=hand_model_path),
    running_mode=VisionRunningMode.VIDEO, 
    num_hands=10,
    min_hand_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)
hand_landmarker = HandLandmarker.create_from_options(hand_options)


# Global Holistic instance (for Single Mode)
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)

def process_frame(frame, thickness, sketch_mode, auto_zoom, single_mode, camera):
    """
    Processes a video frame:
    1. Detects pose using MediaPipe Pose Landmarker (Multi) or Holistic (Single).
    2. Draws stickman.
    """
    if frame is None:
        return None, camera

    # Initialize camera if needed
    if auto_zoom and camera is None:
        camera = StickmanCamera()
    
    # Run inference
    # Resize frame to reduce Gradio latency and inference time
    # Max width 640
    if frame.shape[1] > 640:
        scale = 640 / frame.shape[1]
        new_h = int(frame.shape[0] * scale)
        frame = cv2.resize(frame, (640, new_h))
    
    cam_to_use = camera if auto_zoom else None
    
    # Process
    try:
        # Convert to MP Image (RGB)
        # Note: Gradio image is usually RGB (numpy), but let's confirm.
        # If input type="numpy", it's RGB.
        
        if single_mode:
            # Single Person Mode (Holistic)
            # Holistic expects RGB numpy array
            frame.flags.writeable = False
            results = holistic.process(frame)
            frame.flags.writeable = True # Restore
            
            stickman_img = draw_stickman(
                results, 
                img_shape=frame.shape, 
                thickness=int(thickness), 
                sketch_mode=sketch_mode, 
                camera=cam_to_use,
                mode='single'
            )
            
        else:
            # Multi Person Mode (Pose Landmarker)
            # Convert to MP Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            timestamp_ms = int(time.time() * 1000)
            
            detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)
            hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
            
            stickman_img = draw_stickman(
                detection_result.pose_landmarks, 
                img_shape=frame.shape, 
                thickness=int(thickness), 
                sketch_mode=sketch_mode, 
                camera=cam_to_use,
                mode='multi',
                multi_hand_landmarks=hand_result.hand_landmarks
            )
        
        return stickman_img, camera
    except Exception as e:
        print(f"Error processing frame: {e}")
        # If an error occurs, return a blank white image
        h, w = frame.shape[:2]
        return np.ones((h, w, 3), dtype=np.uint8) * 255, camera

# Gradio Interface
with gr.Blocks(title="YOLO Stickman Motion App") as demo:
    gr.Markdown("# YOLO Stickman Motion App")
    
    with gr.Row():
        input_video = gr.Image(sources=["webcam"], label="Webcam Input", type="numpy")
        output_video = gr.Image(label="Stickman Output", type="numpy")
        
    with gr.Row():
        thickness_slider = gr.Slider(minimum=1, maximum=20, value=4, step=1, label="Stick Thickness")
        sketch_checkbox = gr.Checkbox(label="Handwriting Style", value=False)
        zoom_checkbox = gr.Checkbox(label="Auto Zoom (Tracking)", value=False)
        single_checkbox = gr.Checkbox(label="Single Person Mode (Detailed Fingers)", value=False)

    camera_state = gr.State()

    # Streaming event
    input_video.stream(
        process_frame, 
        inputs=[input_video, thickness_slider, sketch_checkbox, zoom_checkbox, single_checkbox, camera_state], 
        outputs=[output_video, camera_state],
        show_progress=False
    )
    
    # Update on slider change too (optional, but meaningful only if frame creates update)
    # thickness_slider.change(...) # Hard to drive video from slider unless we store last frame.
    
demo.queue()
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
