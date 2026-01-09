import numpy as np

class StickmanCamera:
    """
    Handles auto-zoom and tracking smoothing.
    """
    def __init__(self, smoothing=0.1):
        self.scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.smoothing = smoothing
        self.first_frame = True
        self.facing_right = True # State for face direction hysteresis

    def update(self, landmarks, img_w, img_h):
        if not landmarks: return
        
        # 1. Calculate Bounding Box of Pose
        xs = [lm.x for lm in landmarks.landmark if lm.visibility > 0.5]
        ys = [lm.y for lm in landmarks.landmark if lm.visibility > 0.5]
        
        if not xs or not ys: return

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        # 2. Calculate Target Scale & Center
        # Target: Pose height should be ~80% of screen height
        pose_h = max_y - min_y
        pose_w = max_x - min_x
        
        # Avoid division by zero or extreme zoom
        if pose_h < 0.1: pose_h = 0.1 
        
        target_scale = 0.8 / pose_h
        # Clamp scale
        target_scale = max(0.5, min(target_scale, 3.0))

        # Target center (normalized)
        target_cx = (min_x + max_x) / 2
        target_cy = (min_y + max_y) / 2
        
        # 3. Calculate Target Offset to center the pose
        # We want target_cx to map to 0.5 (center of screen)
        # (target_cx + offset_x) * scale = ... wait.
        # Let's do: ScreenPt = (NormPt - Center) * Scale * Size + ScreenCenter
        # So we just track Center and Scale.
        
        if self.first_frame:
            self.scale = target_scale
            self.offset_x = target_cx
            self.offset_y = target_cy
            self.first_frame = False
        else:
            self.scale += (target_scale - self.scale) * self.smoothing
            self.offset_x += (target_cx - self.offset_x) * self.smoothing
            self.offset_y += (target_cy - self.offset_y) * self.smoothing
            
        # 4. Face Direction Smoothing
        # Indices: Nose=0, LeftShoulder=11, RightShoulder=12
        # We need raw landmarks for this calculation
        if len(landmarks.landmark) > 12:
            nose = landmarks.landmark[0]
            ls = landmarks.landmark[11]
            rs = landmarks.landmark[12]
            
            if nose.visibility > 0.5 and ls.visibility > 0.5 and rs.visibility > 0.5:
                neck_x = (ls.x + rs.x) / 2
                raw_diff = nose.x - neck_x
                
                # Use same smoothing factor or separate one?
                face_smoothing = 0.1
                if not hasattr(self, 'face_diff_smooth'):
                    self.face_diff_smooth = raw_diff
                else:
                    self.face_diff_smooth += (raw_diff - self.face_diff_smooth) * face_smoothing
            elif hasattr(self, 'face_diff_smooth'):
                 # decay to 0 if lost tracking? Or keep last? Keep last is better.
                 pass

    def transform(self, pt_norm, img_w, img_h):
        """
        Transforms a normalized point (0-1) to screen coordinates (px)
        based on current camera state.
        """
        # Center the point relative to the tracked center
        x = (pt_norm[0] - self.offset_x)
        y = (pt_norm[1] - self.offset_y)
        
        # Scale
        x *= self.scale
        y *= self.scale
        
        # Move back to screen center
        # Aspect ratio correction? 
        # Normalized coordinates are usually 0-1, but aspect ratio of image matters for visual squareness.
        # MP normalized coords: x is 0-1 (width), y is 0-1 (height).
        # To keep aspect ratio correct during zoom:
        # We should scale x and y by the same factor relative to pixels.
        
        # Let's project to pixels first assuming identity camera
        px = pt_norm[0] * img_w
        py = pt_norm[1] * img_h
        
        # Tracked center in pixels
        cx = self.offset_x * img_w
        cy = self.offset_y * img_h
        
        # Current point relative to tracked center (pixels)
        dx = px - cx
        dy = py - cy
        
        # Scaled
        dx *= self.scale
        dy *= self.scale
        
        # Screen Center
        scx = img_w / 2
        scy = img_h / 2
        
        final_x = scx + dx
        final_y = scy + dy
        
        return np.array([final_x, final_y])
