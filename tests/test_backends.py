import cv2
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stickman_processor import StickmanProcessor

def test_processor():
    print("Testing Standard StickmanProcessor (Hybrid)...")
    try:
        processor = StickmanProcessor()
        # Create a blank image
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Process
        output, _ = processor.process_frame(img)
        print("Processor Processing Success!")
        return True
    except Exception as e:
        print(f"Processor Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if test_processor():
        print("\nTest passed!")
        sys.exit(0)
    else:
        print("\nTest failed!")
        sys.exit(1)
