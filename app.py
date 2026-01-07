import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np
from stickman import draw_stickman

import torch

# Check for MPS (Apple Silicon) acceleration
device = 'cpu'
if torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'

print(f"Using device: {device}")

# Load YOLOv8 Pose model
# It will download 'yolov8n-pose.pt' on first run
model = YOLO('yolov8n-pose.pt')
# Move to device explicitly
model.to(device)

def process_frame(frame):
    """
    Processes a video frame:
    1. Detects pose using YOLOv8.
    2. detailed keypoints.
    3. Draws stickman.
    """
    if frame is None:
        return None

    # Run inference
    # Resize frame to reduce Gradio latency and inference time
    # Max width 640
    if frame.shape[1] > 640:
        scale = 640 / frame.shape[1]
        new_h = int(frame.shape[0] * scale)
        frame = cv2.resize(frame, (640, new_h))
    
    # imgsz=256 significantly speeds up inference (stickman doesn't need high precision)
    # conf=0.5 filters low confidence preds early
    results = model(frame, device=device, verbose=False, imgsz=256, conf=0.4)
    
    # Get keypoints
    if results[0].keypoints is not None and results[0].keypoints.data.shape[0] > 0:
        persons_keypoints = results[0].keypoints.data.cpu().numpy()
        
        # Draw the first person detected
        keypoints = persons_keypoints[0]
        stickman_img = draw_stickman(keypoints, img_shape=frame.shape)
        return stickman_img
    
    # If no person detected, return blank white image
    h, w, _ = frame.shape
    return np.ones((h, w, 3), dtype=np.uint8) * 255

# Gradio Interface
with gr.Blocks(title="YOLO Stickman") as demo:
    gr.Markdown("# YOLO Stickman Motion Capture")
    with gr.Row():
        with gr.Column():
            input_video = gr.Image(sources=["webcam"], streaming=True, label="Webcam Input", type="numpy")
        with gr.Column():
            output_image = gr.Image(label="Stickman Output", type="numpy")

    input_video.stream(process_frame, inputs=input_video, outputs=output_image)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
