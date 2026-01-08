import gradio as gr
import cv2
import numpy as np
import mediapipe as mp
from stickman import draw_stickman, StickmanCamera

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)

def process_frame(frame, thickness, sketch_mode, auto_zoom, camera):
    """
    Processes a video frame:
    1. Detects pose using MediaPipe Holistic.
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
    
    # Ensure frame is writable for MediaPipe
    frame.flags.writeable = False
    
    # Process
    try:
        results = holistic.process(frame)
        
        # Draw
        # Pass camera only if auto_zoom is enabled
        cam_to_use = camera if auto_zoom else None
        
        stickman_img = draw_stickman(results, img_shape=frame.shape, thickness=int(thickness), sketch_mode=sketch_mode, camera=cam_to_use)
        
        return stickman_img, camera
    except Exception as e:
        print(f"Error processing frame: {e}")
        # If an error occurs, return a blank white image or the original frame
        h, w, _ = frame.shape
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

    camera_state = gr.State()

    # Streaming event
    input_video.stream(
        process_frame, 
        inputs=[input_video, thickness_slider, sketch_checkbox, zoom_checkbox, camera_state], 
        outputs=[output_video, camera_state],
        show_progress=False
    )
    
    # Update on slider change too (optional, but meaningful only if frame creates update)
    # thickness_slider.change(...) # Hard to drive video from slider unless we store last frame.
    
demo.queue()
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
