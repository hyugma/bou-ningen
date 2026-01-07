import cv2
import numpy as np

# Keypoint definitions for YOLOv8 Pose (COCO format)
# 0: Nose, 1: Left Eye, 2: Right Eye, 3: Left Ear, 4: Right Ear
# 5: Left Shoulder, 6: Right Shoulder, 7: Left Elbow, 8: Right Elbow
# 9: Left Wrist, 10: Right Wrist, 11: Left Hip, 12: Right Hip
# 13: Left Knee, 14: Right Knee, 15: Left Ankle, 16: Right Ankle

SKELETON_CONNECTIONS = [
    (5, 7), (7, 9),       # Left Arm
    (6, 8), (8, 10),      # Right Arm
    (11, 13), (13, 15),   # Left Leg
    (12, 14), (14, 16),   # Right Leg
    (5, 6),               # Shoulders
    (11, 12),             # Hips
    (5, 11), (6, 12),     # Torso
    (0, 1), (0, 2),       # Face (approximate)
    (1, 3), (2, 4)        # Ears
]

def draw_stickman(keypoints, img_shape=(480, 640)):
    """
    Draws a stickman based on YOLOv8 keypoints.
    
    Args:
        keypoints (numpy.ndarray): Array of shape (17, 3) or (17, 2). 
                                   [x, y, confidence] or [x, y].
        img_shape (tuple): (height, width) of the output image.
        
    Returns:
        numpy.ndarray: The stickman image (white background, black lines).
    """
    # Create white background
    h, w = img_shape[:2]
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
    
    # Check if keypoints are detected
    if keypoints is None or len(keypoints) == 0:
        return canvas

    # Keypoint indices
    NOSE = 0
    L_SHOULDER, R_SHOULDER = 5, 6
    L_ELBOW, R_ELBOW = 7, 8
    L_WRIST, R_WRIST = 9, 10
    L_HIP, R_HIP = 11, 12
    L_KNEE, R_KNEE = 13, 14
    L_ANKLE, R_ANKLE = 15, 16

    # threshold (lower it slightly to be more responsive/forgiving)
    conf_thresh = 0.4

    def get_point(idx):
        if keypoints[idx][2] > conf_thresh:
            return np.array([keypoints[idx][0], keypoints[idx][1]])
        return None

    def get_midpoint(idx1, idx2):
        p1 = get_point(idx1)
        p2 = get_point(idx2)
        if p1 is not None and p2 is not None:
            return (p1 + p2) / 2
        if p1 is not None: return p1
        if p2 is not None: return p2
        return None

    # Calculate Central Joints
    neck = get_midpoint(L_SHOULDER, R_SHOULDER)
    pelvis = get_midpoint(L_HIP, R_HIP)
    nose = get_point(NOSE)

    # Style Config
    COLOR = (0, 0, 0)
    THICKNESS = 6
    LINETYPE = cv2.LINE_AA

    def draw_rounded_line(p1, p2):
        if p1 is None or p2 is None:
            return
        pt1 = (int(p1[0]), int(p1[1]))
        pt2 = (int(p2[0]), int(p2[1]))
        
        cv2.line(canvas, pt1, pt2, COLOR, THICKNESS, LINETYPE)
        # Draw circles at ends for rounded effect
        cv2.circle(canvas, pt1, int(THICKNESS/2), COLOR, -1, LINETYPE)
        cv2.circle(canvas, pt2, int(THICKNESS/2), COLOR, -1, LINETYPE)

    # 1. Draw Head
    # Connect Nose to Neck? Or just a detached circle?
    # User said: "connect the line of head to the body line"
    if nose is not None and neck is not None:
        # Calculate radius
        dist_shoulder = 0
        p_l_sh = get_point(L_SHOULDER)
        p_r_sh = get_point(R_SHOULDER)
        if p_l_sh is not None and p_r_sh is not None:
            dist_shoulder = np.linalg.norm(p_l_sh - p_r_sh)
        
        radius = 30
        if dist_shoulder > 0:
            radius = int(dist_shoulder / 3.0)
            if radius < 15: radius = 15

        # Refine Head position: 
        # Nose is usually the center of face.
        nose_pt = (int(nose[0]), int(nose[1]))
        cv2.circle(canvas, nose_pt, radius, COLOR, 4, LINETYPE) # Head Outline
        
        # Connect Head (Nose) to Neck
        # To avoid drawing line *inside* the head, we could calculate intersection, 
        # but for simplicity let's draw line from Nose to Neck, behind the circle?
        # Or just Nose to Neck.
        # Actually, let's draw the line first, then the circle (filled with white?)
        # If we draw simple circle, the line inside is visible.
        # Let's simple draw line Nose->Neck.
        draw_rounded_line(nose, neck)
        
        # Redraw Head Circle with white fill to cover the line, then black border
        cv2.circle(canvas, nose_pt, radius, (255, 255, 255), -1, LINETYPE)
        cv2.circle(canvas, nose_pt, radius, COLOR, THICKNESS, LINETYPE)

    elif nose is not None:
         # Fallback if no neck
        cv2.circle(canvas, (int(nose[0]), int(nose[1])), 30, COLOR, THICKNESS, LINETYPE)

    # 2. Draw Body
    if neck is not None and pelvis is not None:
        draw_rounded_line(neck, pelvis)

    # 3. Draw Arms (From Neck)
    # L_Arm: Neck -> L_Elbow -> L_Wrist
    if neck is not None:
        l_elbow = get_point(L_ELBOW)
        l_wrist = get_point(L_WRIST)
        draw_rounded_line(neck, l_elbow)
        draw_rounded_line(l_elbow, l_wrist)
        
        r_elbow = get_point(R_ELBOW)
        r_wrist = get_point(R_WRIST)
        draw_rounded_line(neck, r_elbow)
        draw_rounded_line(r_elbow, r_wrist)

    # 4. Draw Legs (From Pelvis)
    # L_Leg: Pelvis -> L_Knee -> L_Ankle
    if pelvis is not None:
        l_knee = get_point(L_KNEE)
        l_ankle = get_point(L_ANKLE)
        draw_rounded_line(pelvis, l_knee)
        draw_rounded_line(l_knee, l_ankle)

        r_knee = get_point(R_KNEE)
        r_ankle = get_point(R_ANKLE)
        draw_rounded_line(pelvis, r_knee)
        draw_rounded_line(r_knee, r_ankle)

    return canvas
