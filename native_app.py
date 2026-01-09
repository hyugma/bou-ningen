import cv2
import argparse
import numpy as np
import time
from stickman_processor import StickmanProcessor

def main():
    parser = argparse.ArgumentParser(description='Native Stickman Capture')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0).')
    parser.add_argument('--thickness', type=int, default=4, help='Stickman line thickness (default: 4).')
    parser.add_argument('--track', action='store_true', help='Enable auto-tracking (zoom) mode.')
    parser.add_argument('--virtual', action='store_true', help='Enable Output to Virtual Camera (for Zoom).')
    parser.add_argument('--single', action='store_true', help='Enable Single Person Mode (High Detail + Fingers).')
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera}.")
        return
    
    # Virtual Camera Setup
    virtual_cam = None
    if args.virtual:
        try:
            import pyvirtualcam
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            print(f"Starting Virtual Camera: {w}x{h} @ {fps}fps")
            virtual_cam = pyvirtualcam.Camera(width=w, height=h, fps=fps, fmt=pyvirtualcam.PixelFormat.RGB)
            print(f"--> Virtual Camera Ready! Select '{virtual_cam.device}' in Zoom/Teams.")
        except ImportError:
            print("Error: pyvirtualcam not found. Please install with `uv add pyvirtualcam`.")
            return
        except Exception as e:
            print(f"Error starting virtual camera: {e}")
            print("Tip: Make sure you have a virtual camera driver (like OBS Studio) installed.")
            return

    print("Press 'q' to quit.")
    if args.single:
        print("Starting in SINGLE Person Mode (Holistic). Fingers enabled.")
    else:
        print("Starting in MULTI Person Mode (Pose + Hand). Fingers enabled.")

    # --- Processor Logic ---
    processor = StickmanProcessor()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Mirror the frame
        frame = cv2.flip(frame, 1)
        
        # Process Frame
        stickman_img, _ = processor.process_frame(
            frame, 
            thickness=args.thickness, 
            auto_track=args.track, 
            single_mode=args.single
            # internal camera state managed by processor for native app
        )

        # Display window
        cv2.imshow('Stickman Native', stickman_img)
        
        # Output to Virtual Camera
        if virtual_cam:
            frame_rgb = cv2.cvtColor(stickman_img, cv2.COLOR_BGR2RGB)
            # Resize if necessary
            if frame_rgb.shape[1] != virtual_cam.width or frame_rgb.shape[0] != virtual_cam.height:
                frame_rgb = cv2.resize(frame_rgb, (virtual_cam.width, virtual_cam.height))
            virtual_cam.send(frame_rgb)
            virtual_cam.sleep_until_next_frame()
        
        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if virtual_cam:
        virtual_cam.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
