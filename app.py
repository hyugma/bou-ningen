import gradio as gr
import cv2
import numpy as np
import mediapipe as mp
from stickman import draw_stickman, StickmanCamera

from stickman_processor import StickmanProcessor

# Initialize Config
# Gradio creates a new process/state sometimes, so global is okay if persistent.
# StickmanProcessor handles lazy loading.
processor = StickmanProcessor()

def process_frame(frame, thickness, sketch_mode, auto_zoom, single_mode, camera):
    """
    Processes a video frame using StickmanProcessor.
    """
    # StickmanProcessor handles resizing and logic.
    output, new_camera = processor.process_frame(
        frame,
        thickness=thickness, 
        sketch_mode=sketch_mode, 
        auto_zoom=auto_zoom, 
        single_mode=single_mode, 
        camera_instance=camera
    )
    return output, new_camera

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
