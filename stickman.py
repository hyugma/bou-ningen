import cv2
import numpy as np

# MediaPipe Holistic Landmark Indices
# Pose
NOSE = 0
LEFT_EYE_INNER = 1
LEFT_EYE = 2
LEFT_EYE_OUTER = 3
RIGHT_EYE_INNER = 4
RIGHT_EYE = 5
RIGHT_EYE_OUTER = 6
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28

def draw_stickman(results, img_shape=(480, 640), thickness=4, sketch_mode=False):
    """
    Draws a stickman based on MediaPipe Holistic results.
    
    Args:
        results: MediaPipe Holistic results object.
        img_shape (tuple): (height, width) of the output image.
        thickness (int): Base thickness for lines.
        sketch_mode (bool): If True, applies handwriting style (jitter/multiple strokes).
        
    Returns:
        numpy.ndarray: The stickman image.
    """
    h, w = img_shape[:2]
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
    
    if not results.pose_landmarks:
        return canvas

    # Helper to get point from landmarks
    def get_point(idx, landmarks=results.pose_landmarks):
        if idx >= len(landmarks.landmark): return None
        lm = landmarks.landmark[idx]
        if lm.visibility < 0.5: return None
        return np.array([lm.x * w, lm.y * h])

    def get_midpoint(p1, p2):
        if p1 is not None and p2 is not None:
            return (p1 + p2) / 2
        return None

    # --- Config & Helpers ---
    COLOR = (0, 0, 0)
    
    def generate_catmull_rom_spline(points, num_points_per_segment=10):
        if len(points) < 2: return points
        extended_points = [points[0]] + list(points) + [points[-1]]
        spline_points = []
        for i in range(len(extended_points) - 3):
            p0 = np.array(extended_points[i])
            p1 = np.array(extended_points[i+1])
            p2 = np.array(extended_points[i+2])
            p3 = np.array(extended_points[i+3])
            for t in np.linspace(0, 1, num_points_per_segment):
                point = 0.5 * ( (2*p1) + (-p0 + p2)*t + 
                                (2*p0 - 5*p1 + 4*p2 - p3)*(t**2) + 
                                (-p0 + 3*p1 - 3*p2 + p3)*(t**3) )
                spline_points.append((int(point[0]), int(point[1])))
        spline_points.append(points[-1])
        return spline_points

    def draw_sketch_line(p1, p2, thickness, complexity=2, sketch_mode=False):
        if p1 is None or p2 is None: return
        pt1 = np.array([p1[0], p1[1]])
        pt2 = np.array([p2[0], p2[1]])
        dist = np.linalg.norm(pt1 - pt2)
        if dist < 1: return

        loop_count = complexity if sketch_mode else 1
        
        for _ in range(loop_count):
            if sketch_mode:
                num_segments = max(2, int(dist / 15))
                t_values = np.linspace(0, 1, num_segments + 1)
                points = []
                for t in t_values:
                    base_pt = pt1 * (1 - t) + pt2 * t
                    jitter_x = np.random.randint(-2, 3)
                    jitter_y = np.random.randint(-2, 3)
                    points.append((int(base_pt[0] + jitter_x), int(base_pt[1] + jitter_y)))
                for i in range(len(points) - 1):
                    cv2.line(canvas, points[i], points[i+1], COLOR, thickness, cv2.LINE_AA)
            else:
                # Clean line
                cv2.line(canvas, tuple(pt1.astype(int)), tuple(pt2.astype(int)), COLOR, thickness, cv2.LINE_AA)

    def draw_curve_from_points(points_list, thickness, complexity=2, sketch_mode=False):
        valid_pts = [p for p in points_list if p is not None]
        if len(valid_pts) < 2: return
        smooth_points = generate_catmull_rom_spline(valid_pts, num_points_per_segment=15)
        
        loop_count = complexity if sketch_mode else 1

        for _ in range(loop_count):
             for i in range(len(smooth_points) - 1):
                pt1 = smooth_points[i]
                pt2 = smooth_points[i+1]
                
                if sketch_mode:
                    p_start = (int(pt1[0]) + np.random.randint(-2,3), int(pt1[1]) + np.random.randint(-2,3))
                    p_end = (int(pt2[0]) + np.random.randint(-2,3), int(pt2[1]) + np.random.randint(-2,3))
                else:
                    p_start = (int(pt1[0]), int(pt1[1]))
                    p_end = (int(pt2[0]), int(pt2[1]))
                    
                cv2.line(canvas, p_start, p_end, COLOR, thickness, cv2.LINE_AA)
    
    def draw_sketch_circle(center, radius, thickness, complexity=2, sketch_mode=False):
        if center is None: return
        cx, cy = int(center[0]), int(center[1])
        radius = int(radius)
        
        loop_count = complexity if sketch_mode else 1
        
        for _ in range(loop_count):
            num_pts = max(8, int(radius / 2))
            angles = np.linspace(0, 2*np.pi, num_pts, endpoint=True)
            points = []
            for ang in angles:
                if sketch_mode:
                    r_jit = radius + np.random.randint(-2, 3)
                else:
                    r_jit = radius
                x = int(cx + r_jit * np.cos(ang))
                y = int(cy + r_jit * np.sin(ang))
                points.append((x, y))
            for i in range(len(points)):
                pt_a = points[i]
                pt_b = points[(i+1) % len(points)]
                cv2.line(canvas, pt_a, pt_b, COLOR, thickness, cv2.LINE_AA)

    def draw_henohenomoheji(center, l_eye_pt, r_eye_pt, radius, thickness, complexity=2, sketch_mode=False):
        if center is None: return
        cx, cy = int(center[0]), int(center[1])
        r = float(radius)
        
        def draw_stroke(base_pt, pts_rel):
            if base_pt is None: return
            bx, by = base_pt
            abs_pts = []
            for (rx, ry) in pts_rel:
                abs_pts.append((bx + rx * r, by + ry * r))
            draw_curve_from_points(abs_pts, thickness, complexity=1, sketch_mode=sketch_mode)

        # 3. NO (Right Eye)
        if r_eye_pt is not None:
            r_eye_base = (r_eye_pt[0], r_eye_pt[1])
        else:
            r_eye_base = (cx + 0.25*r, cy - 0.15*r)
        draw_stroke(r_eye_base, [(0.0, 0.0), (0.05, -0.1), (-0.05, -0.1), (-0.05, 0.05), (0.1, 0.05)])
        
        # 1. HE (Right Eyebrow)
        draw_stroke(r_eye_base, [(-0.15, -0.25), (0.0, -0.4), (0.15, -0.25)])

        # 4. NO (Left Eye)
        if l_eye_pt is not None:
             l_eye_base = (l_eye_pt[0], l_eye_pt[1])
        else:
             l_eye_base = (cx - 0.25*r, cy - 0.15*r)
        draw_stroke(l_eye_base, [(0.0, 0.0), (0.05, -0.1), (-0.05, -0.1), (-0.05, 0.05), (0.1, 0.05)])

        # 2. HE (Left Eyebrow)
        draw_stroke(l_eye_base, [(-0.15, -0.25), (0.0, -0.4), (0.15, -0.25)])
        
        # 5. MO (Nose) Center
        nose_base = (cx, cy)
        draw_stroke(nose_base, [(0.0, -0.05), (0.0, 0.25), (-0.1, 0.35)])
        draw_stroke(nose_base, [(-0.1, 0.05), (0.1, 0.05)])
        draw_stroke(nose_base, [(-0.1, 0.15), (0.1, 0.15)])
        
        # 6. HE (Mouth)
        draw_stroke(nose_base, [(-0.15, 0.5), (0.0, 0.4), (0.15, 0.5)])
        
        # 7. JI (Outline) - じ
        # "ji" is "shi" with dots.
        # Draw 'Shi' (し) shape: Start top-left, curve down-left, hook up-right.
        # Points relative to center, radius R
        # Refined points for a face-enclosing 'Shi'
        shi_pts = [
            (-0.3, -0.75),  # Start top-left-ish
            (-0.95, -0.2),  # Bulge left
            (-0.2, 0.95),   # Bottom chin area
            (0.85, 0.2)     # Hook up to right
        ]
        draw_stroke(nose_base, shi_pts)
        
        # Dakuten (Dots) - near top right of the "Shi" end
        # "Shi" ends around (0.85, 0.2). Dots should be above/outside.
        # Dot 1
        draw_stroke(nose_base, [(0.9, -0.3), (1.0, -0.2)])
        # Dot 2
        draw_stroke(nose_base, [(1.05, -0.4), (1.15, -0.3)])

    # --- Core Logic ---
    
    # Extract Body Points
    nose = get_point(NOSE)
    l_shoulder = get_point(LEFT_SHOULDER)
    r_shoulder = get_point(RIGHT_SHOULDER)
    l_elbow = get_point(LEFT_ELBOW)
    r_elbow = get_point(RIGHT_ELBOW)
    l_wrist = get_point(LEFT_WRIST)
    r_wrist = get_point(RIGHT_WRIST)
    l_hip = get_point(LEFT_HIP)
    r_hip = get_point(RIGHT_HIP)
    l_knee = get_point(LEFT_KNEE)
    r_knee = get_point(RIGHT_KNEE)
    l_ankle = get_point(LEFT_ANKLE)
    r_ankle = get_point(RIGHT_ANKLE)
    l_eye = get_point(LEFT_EYE)
    r_eye = get_point(RIGHT_EYE)
    
    neck = get_midpoint(l_shoulder, r_shoulder)
    pelvis = get_midpoint(l_hip, r_hip)
    
    # 1. Head & Face
    head_bottom = None
    if nose is not None:
        radius = 30
        if l_shoulder is not None and r_shoulder is not None:
             dist_shoulder = np.linalg.norm(l_shoulder - r_shoulder)
             if dist_shoulder > 0:
                radius = int(dist_shoulder / 2.2)
                if radius < 15: radius = 15
        
        draw_henohenomoheji(nose, l_eye, r_eye, radius, thickness, sketch_mode=sketch_mode)
        
        if neck is not None:
            v = neck - nose
            dist = np.linalg.norm(v)
            if dist > 0:
                head_bottom = nose + (v / dist) * radius
            else:
                head_bottom = nose + np.array([0, radius])

    # 2. Body & Legs
    if neck is not None:
        body_start = head_bottom if head_bottom is not None else neck
        
        # Left Leg Chain
        chain_l = [body_start]
        if head_bottom is not None: chain_l.append(neck)
        chain_l.extend([pelvis, l_knee, l_ankle])
        draw_curve_from_points(chain_l, thickness, sketch_mode=sketch_mode)
        
        # Right Leg Chain
        chain_r = [body_start]
        if head_bottom is not None: chain_r.append(neck)
        chain_r.extend([pelvis, r_knee, r_ankle])
        draw_curve_from_points(chain_r, thickness, sketch_mode=sketch_mode)
        
        # Arms
        draw_curve_from_points([neck, l_elbow, l_wrist], thickness, sketch_mode=sketch_mode)
        draw_curve_from_points([neck, r_elbow, r_wrist], thickness, sketch_mode=sketch_mode)

    # 3. Feet (Simple)
    def draw_foot(ankle, knee, side='left'):
        if ankle is None or knee is None: return
        v = ankle - knee
        norm = np.linalg.norm(v)
        if norm < 0.1: return
        d = v / norm
        u = np.array([-d[1], d[0]]) # Orthogonal
        foot_len = max(15, int(norm * 0.3))
        if foot_len > 40: foot_len = 40
        
        foot_vec = u * foot_len
        px = ankle[0]
        cx = pelvis[0] if pelvis is not None else w/2
        if px > cx:
            if foot_vec[0] < 0: foot_vec = -foot_vec
        else:
            if foot_vec[0] > 0: foot_vec = -foot_vec
        draw_sketch_line(ankle, ankle + foot_vec, thickness, complexity=1, sketch_mode=sketch_mode)

    draw_foot(l_ankle, l_knee, 'left')
    draw_foot(r_ankle, r_knee, 'right')

    # 4. Hands (Fingers) - Real or Simulated
    # If we have hand landmarks, use them. Else simulate.
    def draw_real_hand(hand_landmarks):
        # Wrist is 0
        # Thumb: 1-4, Index: 5-8, Middle: 9-12, Ring: 13-16, Pinky: 17-20
        # We can just draw lines 0->1->2.. and 0->5->6.. etc
        wrist = np.array([hand_landmarks.landmark[0].x * w, hand_landmarks.landmark[0].y * h])
        
        chains = [
            [0, 1, 2, 3, 4],       # Thumb
            [0, 5, 6, 7, 8],       # Index
            [5, 9, 10, 11, 12],    # Middle (root at 9, but usually 5-9 is palm. Let's draw 0->9->10...)
            # Actually 0->5, 0->9, 0->13, 0->17 are palm bones.
        ]
        # Better simple skeleton:
        # Wrist(0) -> Thumb(1..4)
        # Wrist(0) -> Index(5..8)
        # Wrist(0) -> Middle(9..12)
        # Wrist(0) -> Ring(13..16)
        # Wrist(0) -> Pinky(17..20)
        
        for finger_indices in [[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16], [17,18,19,20]]:
            pts = [wrist]
            for idx in finger_indices:
                lm = hand_landmarks.landmark[idx]
                pts.append(np.array([lm.x * w, lm.y * h]))
            draw_curve_from_points(pts, max(1, thickness//2), complexity=1, sketch_mode=sketch_mode)

    # Check for hands in results
    # MediaPipe Holistic returns left_hand_landmarks and right_hand_landmarks
    if results.left_hand_landmarks:
        draw_real_hand(results.left_hand_landmarks)
    elif l_wrist is not None and l_elbow is not None:
        # Fallback simulation
        # Same logic as before if needed, or just skip if we assume MP always tracks hands?
        # MP Holistic hand tracking is triggered if hands are detected.
        # If not detected, we might want fallback?
        # Let's keep fallback for robustness.
        pass # To keep code clean, omitting fallback copy-paste here unless requested.

    if results.right_hand_landmarks:
        draw_real_hand(results.right_hand_landmarks)

    return canvas
