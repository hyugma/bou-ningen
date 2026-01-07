import cv2
import numpy as np
import stickman

# Mock Keypoints for a standing pose
# [x, y, conf]
# Center is approx (240, 320)
mock_keypoints = np.zeros((17, 3))

# Head/Torso
mock_keypoints[0] = [240, 100, 0.9] # Nose
mock_keypoints[5] = [200, 150, 0.9] # L Shoulder
mock_keypoints[6] = [280, 150, 0.9] # R Shoulder
mock_keypoints[11] = [210, 300, 0.9] # L Hip
mock_keypoints[12] = [270, 300, 0.9] # R Hip

# Arms (Waving)
mock_keypoints[7] = [160, 200, 0.9] # L Elbow
mock_keypoints[9] = [120, 120, 0.9] # L Wrist
mock_keypoints[8] = [320, 200, 0.9] # R Elbow
mock_keypoints[10] = [360, 250, 0.9] # R Wrist

# Legs
mock_keypoints[13] = [210, 400, 0.9] # L Knee
mock_keypoints[15] = [210, 500, 0.9] # L Ankle
mock_keypoints[14] = [270, 400, 0.9] # R Knee
mock_keypoints[16] = [270, 500, 0.9] # R Ankle

# Generate Image
img = stickman.draw_stickman(mock_keypoints)

# Save
cv2.imwrite("verification_stickman.png", img)
print("Saved verification_stickman.png")
