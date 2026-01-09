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

from stickman_camera import StickmanCamera

class MockHand:
    """Helper to wrap a list of landmarks into an object with .landmark property."""
    def __init__(self, lms):
        self.landmark = lms

class MockLandmarks:
    """Helper to wrap a list of landmarks into an object with .landmark property."""
    def __init__(self, lms):
        self.landmark = lms


def draw_single_person(canvas, landmarks, img_shape, thickness, camera, left_hand=None, right_hand=None):
    """
    Draws a single stickman from a list of Pose Landmarker landmarks.
    Optional: left_hand, right_hand (MediaPipe landmarks) for detailed fingers.
    """
    h, w = img_shape[:2]
    # Canvas is passed in
    
    if not landmarks:
        return



    # Helper to get point from landmarks
    def get_point(idx, landmarks=landmarks):
        if idx >= len(landmarks): return None
        lm = landmarks[idx]
        if lm.visibility < 0.5: return None
        
        pt_norm = np.array([lm.x, lm.y])
        
        if camera:
            return camera.transform(pt_norm, w, h)
        else:
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

    def draw_line(p1, p2, thickness):
        if p1 is None or p2 is None: return
        pt1 = np.array([p1[0], p1[1]])
        pt2 = np.array([p2[0], p2[1]])
        dist = np.linalg.norm(pt1 - pt2)
        if dist < 1: return

        # Clean line
        cv2.line(canvas, tuple(pt1.astype(int)), tuple(pt2.astype(int)), COLOR, thickness, cv2.LINE_AA)

    def draw_curve_from_points(points_list, thickness):
        valid_pts = [p for p in points_list if p is not None]
        if len(valid_pts) < 2: return
        smooth_points = generate_catmull_rom_spline(valid_pts, num_points_per_segment=15)
        
        for i in range(len(smooth_points) - 1):
            pt1 = smooth_points[i]
            pt2 = smooth_points[i+1]
            
            p_start = (int(pt1[0]), int(pt1[1]))
            p_end = (int(pt2[0]), int(pt2[1]))
            
            cv2.line(canvas, p_start, p_end, COLOR, thickness, cv2.LINE_AA)
    
    def draw_sketch_circle(center, radius, thickness):
        if center is None: return
        cx, cy = int(center[0]), int(center[1])
        radius = int(radius)
        
        num_pts = max(8, int(radius / 2))
        angles = np.linspace(0, 2*np.pi, num_pts, endpoint=True)
        points = []
        for ang in angles:
            r_jit = radius
            x = int(cx + r_jit * np.cos(ang))
            y = int(cy + r_jit * np.sin(ang))
            points.append((x, y))
        for i in range(len(points)):
            pt_a = points[i]
            pt_b = points[(i+1) % len(points)]
            cv2.line(canvas, pt_a, pt_b, COLOR, thickness, cv2.LINE_AA)

    def draw_henohenomoheji(center, l_eye_pt, r_eye_pt, radius, thickness, facing_right=True):
        if center is None: return
        cx, cy = int(center[0]), int(center[1])
        r = float(radius)
        
        def draw_stroke(base_pt, pts_rel):
            if base_pt is None: return
            bx, by = base_pt
            abs_pts = []
            for (rx, ry) in pts_rel:
                abs_pts.append((bx + rx * r, by + ry * r))
            draw_curve_from_points(abs_pts, thickness)

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
        # Standard "Shi" (し) curves from top-left, down, to right.
        # shape: Start top-left, curve down-left, hook up-right.
        
        # Default shape (Facing RIGHT, i.e. Screen Right)
        # In this context, "Facing Right" means the face opens to the right.
        # The vertical stroke of 'Shi' is on the left (Back of head).
        # The hook is on the right (Chin/Face).
        
        # If Facing LEFT (Screen Left):
        # We should flip it horizontally.
        # Vertical stroke on Right (Back of head).
        # Hook on Left (Chin/Face).
        
        # x_mult = 1.0 if facing_right else -1.0
        x_mult = -1.0 if facing_right else 1.0
        
        # ORIRINAL
        #shi_pts = [
        #    (-0.3 * x_mult, -0.75),  # Start top-back
        #    (-0.95 * x_mult, -0.2),  # Bulge back
        #    (-0.2 * x_mult, 0.95),   # Bottom chin area
        #    (0.85 * x_mult, 0.2)     # Hook up to face front
        #]

        shi_pts = [
            (-0.7 * x_mult, -0.75),  # Start top-back
            (-0.7 * x_mult, 0.5),    # Bulge back
            (-0.1 * x_mult, 0.95),   # Bottom chin area
            (0.85 * x_mult, 0.2)     # Hook up to face front
        ]

        draw_stroke(nose_base, shi_pts)
        
        # Dakuten (Dots)
        # User requested: "dot to be the opposite side of the facing direction"
        # If facing Right -> Dots on Left (Back).
        # If facing Left -> Dots on Right (Back).
        # Our X flip logic above puts the "back" (vertical stroke) on the left if facing Right.
        # So we want dots on the Left if facing Right.
        # Wait, standard `じ` dots are on the Top-Right (Open side).
        # But user wants "Opposite of facing".
        # If Facing Right -> Dots on Left (Back).
        # If Facing Left -> Dots on Right (Back).
        
        # Let's define dots relative to "Back" of head.
        # Back is where x is negative (if facing Right).
        # So dots should be around x = -0.something.
        
        # Dot 1 (-0.9, -0.3) if facing Right?
        # Let's try to place them "Behind" the vertical stroke.
        # Vertical stroke is around x=-0.3 to -0.95.
        
        # User requested: "dots need to come to the open part of the circle (head edge)"
        # This means the Face side (Right if facing Right).
        # Previously I put them on the Back side (-x).
        # Now we want them on the Front side (+x).
        
        dot_x_base = 1.0 * x_mult
        
        # Dot 1: Along the edge, probably upper area or near the "ear" position?
        # Standard Ji: Top-Right.
        # Let's place them at x=1.0, y=-0.3 range.
        
        draw_stroke(nose_base, [(dot_x_base, -0.3), (dot_x_base + 0.1 * x_mult, -0.2)])
        # Dot 2
        draw_stroke(nose_base, [(dot_x_base + 0.05 * x_mult, -0.45), (dot_x_base + 0.15 * x_mult, -0.35)])

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
             # Calculate shoulder distance in pixel coords
             # Since points are already transformed by get_point, 
             # dist_shoulder will be in screen pixels (zoomed or not).
             dist_shoulder = np.linalg.norm(l_shoulder - r_shoulder)
             if dist_shoulder > 0:
                radius = int(dist_shoulder / 2.2)
                if radius < 15: radius = 15
        
        facing_right = True
        if neck is not None:
             # Check X direction relative to neck (center of shoulders)
             # Pixels increase right.
             # nose[0] < neck[0] => Nose is Left of Neck => Facing Left.
             
             # Hysteresis Threshold
             # We only switch if we cross the threshold significantly.
             threshold = radius * 0.3 # 30% of head size
             
             # Use smoothed value from camera if available
             if camera and hasattr(camera, 'face_diff_smooth'):
                 # Need to scale normalized diff to pixel space (roughly)
                 # StickmanCamera calculates stats in normalized space (0-1).
                 # Radius is in pixels. Normalized diff is small.
                 # We need to convert normalized diff to pixels: diff * width
                 # But width (w) is available here.
                 diff = camera.face_diff_smooth * w
             else:
                 diff = nose[0] - neck[0]
             
             if diff < -threshold:
                 facing_right = False
             elif diff > threshold:
                 facing_right = True
             else:
                 # Inside deadzone
                 if camera:
                     facing_right = camera.facing_right
                 else:
                     # Default if no camera state
                     facing_right = (diff > 0)

             if camera:
                 camera.facing_right = facing_right
                  
        draw_henohenomoheji(nose, l_eye, r_eye, radius, thickness, facing_right=facing_right)
        
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
        draw_curve_from_points(chain_l, thickness)
        
        # Right Leg Chain
        chain_r = [body_start]
        if head_bottom is not None: chain_r.append(neck)
        chain_r.extend([pelvis, r_knee, r_ankle])
        draw_curve_from_points(chain_r, thickness)
        
        # Arms
        draw_curve_from_points([neck, l_elbow, l_wrist], thickness)
        draw_curve_from_points([neck, r_elbow, r_wrist], thickness)

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
        # Need re-calculated width/center for foot direction logic?
        # Transform logic handles coordinate space transparency.
        # But we need "screen center" logic for foot direction? Or relative to pelvis?
        # Current logic: `if px > cx`. cx is pelvis x or w/2.
        # Since points are transformed, pelvis x is correct relative to ankle x.
        # Logic holds.
        cx = pelvis[0] if pelvis is not None else w/2
        if px > cx:
            if foot_vec[0] < 0: foot_vec = -foot_vec
        else:
            if foot_vec[0] > 0: foot_vec = -foot_vec
        draw_line(ankle, ankle + foot_vec, thickness)

    draw_foot(l_ankle, l_knee, 'left')
    draw_foot(r_ankle, r_knee, 'right')

    # 4. Hands (Fingers) - Real or Simulated
    def draw_real_hand(hand_landmarks):
        # Wrist is 0
        if not hand_landmarks: return
        # Helper to get hand point relative to camera
        def get_hand_pt(idx):
             lm = hand_landmarks.landmark[idx]
             pt_norm = np.array([lm.x, lm.y])
             if camera: return camera.transform(pt_norm, w, h)
             else: return np.array([lm.x * w, lm.y * h])

        wrist = get_hand_pt(0)
        # Segments: Thumb(1-4), Index(5-8), Middle(9-12), Ring(13-16), Pinky(17-20)
        for finger_indices in [[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16], [17,18,19,20]]:
            pts = [wrist]
            for idx in finger_indices: pts.append(get_hand_pt(idx))
            draw_curve_from_points(pts, max(1, thickness//2))

    if left_hand:
        draw_real_hand(left_hand)
    elif l_wrist is not None and l_elbow is not None:
         # Fallback Simple Hand
         hand_rad = int(max(5, thickness * 1.5))
         cx, cy = int(l_wrist[0]), int(l_wrist[1])
         for _ in range(1):
             cv2.circle(canvas, (cx, cy), hand_rad, COLOR, thickness//2)
             
    if right_hand:
         draw_real_hand(right_hand)
    elif r_wrist is not None and r_elbow is not None:
         # Fallback Simple Hand
         hand_rad = int(max(5, thickness * 1.5))
         cx, cy = int(r_wrist[0]), int(r_wrist[1])
         for _ in range(1):
             cv2.circle(canvas, (cx, cy), hand_rad, COLOR, thickness//2)

    return canvas

def draw_stickman(data, img_shape=(480, 640), thickness=4, camera=None, mode='multi', multi_hand_landmarks=None):
    """
    Draws stickman/men.
    Args:
        data: If mode='multi', list of landmark lists (Pose).
              If mode='single', MediaPipe Holistic results object.
        multi_hand_landmarks: List of landmark lists (Hand) for mode='multi'.
    """
    h, w = img_shape[:2]
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
    
    if not data:
        return canvas

    if mode == 'multi':
        # data is multi_pose_landmarks
        multi_pose_landmarks = data
        if not multi_pose_landmarks: return canvas
        
        if camera:
            all_lms = []
            for person_lms in multi_pose_landmarks:
                all_lms.extend(person_lms)
            camera.update(MockLandmarks(all_lms), w, h)

        # Helper to find matching hand
        def find_hand(wrist_pt):
            if not multi_hand_landmarks or wrist_pt is None: return None
            # wrist_pt is screen coordinates if camera, else pixels.
            # But hand landmarks are normalized. 
            # We need to compare them in same space.
            # Let's use normalized coordinates for matching to be robust?
            # actually draw_single_person calculates wrist_pt using camera transform.
            # We should probably get normalized wrist from pose landmark directly.
            return None
        
        # New matching logic:
        # We need normalized wrist for matching.
        
        for landmarks in multi_pose_landmarks:
            # Find closest hands
            l_wrist_idx = 15
            r_wrist_idx = 16
            
            l_hand_match = None
            r_hand_match = None
            
            if multi_hand_landmarks: 

                # Get normalized wrists
                l_wrist_lm = landmarks[l_wrist_idx]
                r_wrist_lm = landmarks[r_wrist_idx]
                
                # Simple Threshold matching (in normalized space)
                thresh = 0.2 # Increased threshold slightly
                
                min_dist_l = thresh
                for hand_lms in multi_hand_landmarks:
                    # Hand wrist is 0
                    h_wrist = hand_lms[0]
                    dist = np.sqrt((l_wrist_lm.x - h_wrist.x)**2 + (l_wrist_lm.y - h_wrist.y)**2)
                    if dist < min_dist_l:
                        min_dist_l = dist
                        l_hand_match = MockHand(hand_lms)

                min_dist_r = thresh
                for hand_lms in multi_hand_landmarks:
                    h_wrist = hand_lms[0]
                    dist = np.sqrt((r_wrist_lm.x - h_wrist.x)**2 + (r_wrist_lm.y - h_wrist.y)**2)
                    if dist < min_dist_r:
                        min_dist_r = dist
                        r_hand_match = MockHand(hand_lms)

            draw_single_person(canvas, landmarks, img_shape, thickness, camera, 
                               left_hand=l_hand_match, right_hand=r_hand_match)

    elif mode == 'single':
        # data is holistic results
        results = data
        if not results.pose_landmarks: return canvas
        
        if camera:
            camera.update(results.pose_landmarks, w, h)
            
        draw_single_person(
            canvas, 
            results.pose_landmarks.landmark, 
            img_shape, 
            thickness, 
            camera,
            left_hand=results.left_hand_landmarks,
            right_hand=results.right_hand_landmarks
        )
        
    return canvas
