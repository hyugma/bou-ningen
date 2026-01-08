import cv2
import argparse
import mediapipe as mp
import numpy as np
import time
from stickman import draw_stickman, StickmanCamera

# Debug MediaPipe
# print("MediaPipe file:", mp.__file__)
# print("MediaPipe dir:", dir(mp))

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def main():
    parser = argparse.ArgumentParser(description='Native Stickman Capture')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0).')
    parser.add_argument('--thickness', type=int, default=4, help='Stickman line thickness (default: 4).')
    parser.add_argument('--sketch', action='store_true', help='Enable handwriting sketch style.')
    parser.add_argument('--zoom', action='store_true', help='Enable auto-zoom (tracking) mode.')
    parser.add_argument('--virtual', action='store_true', help='Enable Output to Virtual Camera (for Zoom).')
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera}.")
        return
    
    # Initialize Camera logic if zoom is enabled
    stickman_camera = StickmanCamera() if args.zoom else None

    # Virtual Camera Setup
    virtual_cam = None
    if args.virtual:
        try:
            import pyvirtualcam
            # Get camera resolution
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            print(f"Starting Virtual Camera: {w}x{h} @ {fps}fps")
            virtual_cam = pyvirtualcam.Camera(width=w, height=h, fps=fps)
            print(f"--> Virtual Camera Ready! Select '{virtual_cam.device}' in Zoom/Teams.")
        except ImportError:
            print("Error: pyvirtualcam not found. Please install with `uv add pyvirtualcam`.")
            return
        except Exception as e:
            print(f"Error starting virtual camera: {e}")
            print("Tip: Make sure you have a virtual camera driver (like OBS Studio) installed.")
            return

    print("Press 'q' to quit.")
    
    # MediaPipe Holistic context
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            # Mirror the frame
            frame = cv2.flip(frame, 1)
            
            # Convert BGR to RGB for MediaPipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Process with MediaPipe
            results = holistic.process(image)
            
            # Draw Stickman
            # draw_stickman returns BGR
            stickman_img = draw_stickman(results, img_shape=frame.shape, thickness=args.thickness, sketch_mode=args.sketch, camera=stickman_camera)

            # Display window
            cv2.imshow('Stickman Native', stickman_img)
            
            # Output to Virtual Camera
            if virtual_cam:
                # pyvirtualcam expects RGB (or RGBA)
                frame_rgb = cv2.cvtColor(stickman_img, cv2.COLOR_BGR2RGB)
                
                # Resize if necessary to match virtual camera setup (e.g. if cam init reported diff size)
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
