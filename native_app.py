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


def main():
    parser = argparse.ArgumentParser(description='Native Stickman Capture')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0).')
    parser.add_argument('--thickness', type=int, default=4, help='Stickman line thickness (default: 4).')
    parser.add_argument('--sketch', action='store_true', help='Enable handwriting sketch style.')
    parser.add_argument('--zoom', action='store_true', help='Enable auto-zoom (tracking) mode.')
    parser.add_argument('--virtual', action='store_true', help='Enable Output to Virtual Camera (for Zoom).')
    parser.add_argument('--single', action='store_true', help='Enable Single Person Mode (High Detail + Fingers).')
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera}.")
        return
    
    # Initialize Camera logic if zoom is enabled
    stickman_camera = StickmanCamera() if args.zoom else None

    # Virtual Camera Setup (Previous code...)
    virtual_cam = None
    if args.virtual:
        try:
            import pyvirtualcam
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
    
    # --- Mode Selection ---
    if args.single:
        print("Starting in SINGLE Person Mode (Holistic). Fingers enabled.")
        mp_holistic = mp.solutions.holistic
        
        with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as holistic:
            
            while True:
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.flip(frame, 1)
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                
                # Holistic Process
                results = holistic.process(image)
                
                # Draw (Single Mode)
                stickman_img = draw_stickman(
                    results, 
                    img_shape=frame.shape, 
                    thickness=args.thickness, 
                    sketch_mode=args.sketch, 
                    camera=stickman_camera,
                    mode='single'
                )

                cv2.imshow('Stickman Native', stickman_img)
                
                if virtual_cam:
                    frame_rgb = cv2.cvtColor(stickman_img, cv2.COLOR_BGR2RGB)
                    if frame_rgb.shape[1] != virtual_cam.width or frame_rgb.shape[0] != virtual_cam.height:
                        frame_rgb = cv2.resize(frame_rgb, (virtual_cam.width, virtual_cam.height))
                    virtual_cam.send(frame_rgb)
                    virtual_cam.sleep_until_next_frame()
                
                if cv2.waitKey(1) & 0xFF == ord('q'): break

    else:
        print("Starting in MULTI Person Mode (Pose Landmarker). No fingers.")
        # MediaPipe Pose Landmarker Setup
        model_path = 'pose_landmarker_full.task'
        import os
        if not os.path.exists(model_path):
            print(f"Model file {model_path} not found. Downloading...")
            import urllib.request
            url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
            try:
                urllib.request.urlretrieve(url, model_path)
                print("Download successful.")
            except Exception as e:
                print(f"Failed to download model: {e}")
                return

        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO, 
            num_poses=5,  # Up to 5 people
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False
        )

        # MediaPipe Hand Landmarker Setup
        hand_model_path = 'hand_landmarker.task'
        if not os.path.exists(hand_model_path):
            print(f"Model file {hand_model_path} not found. Downloading...")
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            try:
                import urllib.request
                urllib.request.urlretrieve(url, hand_model_path)
            except Exception as e:
                print(f"Failed to download hand model: {e}")

        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

        hand_options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=hand_model_path),
            running_mode=VisionRunningMode.VIDEO, 
            num_hands=10,  # 5 people * 2 hands
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
        with PoseLandmarker.create_from_options(options) as landmarker, \
             HandLandmarker.create_from_options(hand_options) as hand_landmarker:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame.")
                    break

                # Mirror the frame
                frame = cv2.flip(frame, 1)
                
                # Convert BGR to RGB for MediaPipe
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
                
                # Process with MediaPipe
                # Use timestamp for VIDEO mode
                timestamp_ms = int(time.time() * 1000)
                detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)
                hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
                
                # Draw Stickman
                # draw_stickman expects a list of landmarks lists
                stickman_img = draw_stickman(
                    detection_result.pose_landmarks, 
                    img_shape=frame.shape, 
                    thickness=args.thickness, 
                    sketch_mode=args.sketch, 
                    camera=stickman_camera,
                    mode='multi',
                    multi_hand_landmarks=hand_result.hand_landmarks
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
