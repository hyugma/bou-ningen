import cv2
from ultralytics import YOLO
import torch
import numpy as np
from stickman import draw_stickman

# Check for MPS (Apple Silicon) acceleration
device = 'cpu'
if torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'

print(f"Using device: {device}")

# Load model
model = YOLO('yolov8n-pose.pt')
model.to(device)

import argparse

def main():
    parser = argparse.ArgumentParser(description='Native Stickman Capture')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0). Try 1 or 2 for iPhone/Continuity Camera.')
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    
    # Set webcam resolution (optional, helps speed if lower)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Starting native stickman capture. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            print("Troubleshooting (macOS):")
            print("1. Ensure your Terminal/IDE has permission to access the Camera.")
            print("   Go to System Settings > Privacy & Security > Camera.")
            print("2. Try running 'tccutil reset Camera' in terminal to reset permissions.")
            break

        # Inference
        results = model(frame, device=device, verbose=False, imgsz=256, conf=0.4)
        
        # Draw Stickman
        stickman_img = None
        if results[0].keypoints is not None and results[0].keypoints.data.shape[0] > 0:
            persons_keypoints = results[0].keypoints.data.cpu().numpy()
            # Draw first person
            stickman_img = draw_stickman(persons_keypoints[0], img_shape=frame.shape)
        else:
            stickman_img = np.ones_like(frame) * 255

        # Show Output
        cv2.imshow('Stickman Realtime', stickman_img)
        
        # Also show original?
        # cv2.imshow('Original', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
