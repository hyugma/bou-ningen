import cv2
import argparse
import mediapipe as mp
import numpy as np
import time
from stickman import draw_stickman

# Debug MediaPipe
print("MediaPipe file:", mp.__file__)
print("MediaPipe dir:", dir(mp))

# Initialize MediaPipe Holistic
try:
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
except AttributeError as e:
    print(f"Error accessing mp.solutions: {e}")
    # Try importing directly
    import mediapipe.python.solutions.holistic
    print("Direct import successful?")
    raise e

def main():
    parser = argparse.ArgumentParser(description='Native Stickman Capture')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0).')
    parser.add_argument('--thickness', type=int, default=4, help='Stickman line thickness (default: 4).')
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera}.")
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
            # draw_stickman now expects 'results' and 'img_shape'
            stickman_img = draw_stickman(results, img_shape=frame.shape, thickness=args.thickness)

            # Display
            cv2.imshow('Stickman Native', stickman_img)
            
            # Quit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
