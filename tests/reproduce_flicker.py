
import sys
import os
import math

# Add parent directory to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from smoother import LandmarkSmoother

# Mock Landmark class
class MockLandmark:
    def __init__(self, x, y, z=0, visibility=1.0):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility

def test_smoothing_jump():
    print("Testing Smoothing Jump...")
    
    # Initialize smoother with alpha=0.5 (heavy smoothing)
    smoother = LandmarkSmoother(alpha=0.5)
    
    # Frame 1: Person A at x=0.2
    # Pose index 0
    pose_a = [MockLandmark(0.2, 0.5)]
    
    print(f"Frame 1 Input: x={pose_a[0].x}")
    smoothed_1 = smoother.update(pose_a)
    print(f"Frame 1 Output: x={smoothed_1[0].x:.4f}")
    
    # Check if initialized correctly
    assert abs(smoothed_1[0].x - 0.2) < 0.001, "Frame 1 should be instant (init)"
    
    # Frame 2: Person B at x=0.8 (Simulating index swap where tracker 0 is now Person B)
    pose_b = [MockLandmark(0.8, 0.5)]
    
    print(f"Frame 2 Input: x={pose_b[0].x} (Large Jump)")
    smoothed_2 = smoother.update(pose_b)
    print(f"Frame 2 Output: x={smoothed_2[0].x:.4f}")
    
    # WITHOUT FIX:
    # Expected: 0.5 * 0.8 + 0.5 * 0.2 = 0.4 + 0.1 = 0.5
    # The output 0.5 is halfway between, creating the "flicker" / "connecting line" artifact.
    
    # WITH FIX (Target):
    # Expected: Should detect jump and reset. Output should be 0.8.
    
    if abs(smoothed_2[0].x - 0.5) < 0.1:
        print("RESULT: SMOOTHING HAPPENED (Artifact exists)")
        return False
    elif abs(smoothed_2[0].x - 0.8) < 0.001:
        print("RESULT: JUMP DETECTED (Artifact fixed)")
        return True
    else:
        print(f"RESULT: Unexpected value {smoothed_2[0].x}")
        return False

if __name__ == "__main__":
    success = test_smoothing_jump()
    if success:
        print("Test PASSED: Jump was handled correctly.")
        sys.exit(0)
    else:
        print("Test FAILED: Interpolation occurred.")
        sys.exit(1)
