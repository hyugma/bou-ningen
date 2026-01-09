class SmoothedLandmark:
    def __init__(self, x=0, y=0, z=0, visibility=0):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility

class LandmarkSmoother:
    def __init__(self, alpha=0.5, visibility_threshold=0.5):
        """
        Args:
            alpha: Smoothing factor (0 < alpha <= 1). Lower = more smoothing, Higher = more responsive.
            visibility_threshold: Minimum visibility to consider a landmark valid for update.
        """
        self.alpha = alpha
        self.visibility_threshold = visibility_threshold
        self.smoothed_landmarks = {} # index -> SmoothedLandmark

    def update(self, landmarks):
        """
        Updates internal state with new landmarks and returns smoothed list.
        Args:
            landmarks: List of landmark objects (normalized).
        Returns:
            List of objects compatible with MediaPipe landmark interface (.x, .y, .z, .visibility).
        """
        if not landmarks:
            return []

        result = []
        
        # Determine input type (list of objects or object with .landmark)
        input_lms = landmarks
        if hasattr(landmarks, 'landmark'):
             input_lms = landmarks.landmark

        for i, lm in enumerate(input_lms):
            if i not in self.smoothed_landmarks:
                # Initialize
                self.smoothed_landmarks[i] = SmoothedLandmark(lm.x, lm.y, lm.z, lm.visibility)
            else:
                prev = self.smoothed_landmarks[i]
                
                # Only update if visibility is good? 
                # Or always update but trust low visibility less?
                # Simple EMA on coordinates.
                
                # Adaptive alpha based on velocity could be complex. Sticking to simple EMA.
                
                # If current observation is bad, maybe don't update or update slowly?
                # For stickman, if MediaPipe loses track, it flickers.
                # If visibility drops, we might want to hold last position or decay.
                
                if lm.visibility >= self.visibility_threshold:
                    prev.x = self.alpha * lm.x + (1 - self.alpha) * prev.x
                    prev.y = self.alpha * lm.y + (1 - self.alpha) * prev.y
                    prev.z = self.alpha * lm.z + (1 - self.alpha) * prev.z
                    prev.visibility = self.alpha * lm.visibility + (1 - self.alpha) * prev.visibility
                else:
                    # If visibility is low, decay visibility but keep position?
                    # Or just trust the model's low visibility?
                    prev.visibility = lm.visibility
                    # Optionally drift position? stick to last known good?
                    # Let's trust model position but smoothed.
                    prev.x = self.alpha * lm.x + (1 - self.alpha) * prev.x
                    prev.y = self.alpha * lm.y + (1 - self.alpha) * prev.y
            
            # Create a NEW object for result to avoid reference issues
            s = self.smoothed_landmarks[i]
            result.append(SmoothedLandmark(s.x, s.y, s.z, s.visibility))
            
        return result
