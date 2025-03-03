import cv2
import mediapipe as mp
import math

class HandDetector:
    def __init__(self, mode=False, max_hands=2, detection_confidence=0.5, tracking_confidence=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence
        )
        # Drawing utilities
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
    def find_hands(self, img, draw=True):
        """Process image and detect hands."""
        # Convert to RGB (MediaPipe requires RGB input)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process the image
        self.results = self.hands.process(img_rgb)
        
        # Draw hand landmarks if detected
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    # Draw landmarks and connections
                    self.mp_draw.draw_landmarks(
                        img, 
                        hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
        
        return img
    
    def find_position(self, img, hand_no=0, draw=False):
        """Find positions of hand landmarks."""
        landmark_list = []
        h, w, c = img.shape
        
        if self.results.multi_hand_landmarks:
            # Get the hand (if there are multiple hands)
            if len(self.results.multi_hand_landmarks) > hand_no:
                hand = self.results.multi_hand_landmarks[hand_no]
                
                # Extract landmark positions
                for id, lm in enumerate(hand.landmark):
                    # Convert normalized coordinates to pixel coordinates
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmark_list.append([id, cx, cy])
                    
                    # Draw circles at landmarks if requested
                    if draw:
                        cv2.circle(img, (cx, cy), 7, (255, 0, 0), cv2.FILLED)
        
        return landmark_list
    
    def fingers_up(self, landmark_list):
        """Determine which fingers are up."""
        fingers = []
        
        # Check if we have enough landmarks
        if not landmark_list or len(landmark_list) < 21:
            return [0, 0, 0, 0, 0]
        
        # Finger tip IDs
        tip_ids = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky
        
        # Thumb (special case - compare with thumb base)
        # Check if thumb is to the right of the thumb base for right hand
        if landmark_list[tip_ids[0]][1] > landmark_list[tip_ids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        
        # 4 fingers
        for id in range(1, 5):
            # Check if fingertip y is lower than middle joint y
            if landmark_list[tip_ids[id]][2] < landmark_list[tip_ids[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        
        return fingers
    
    def find_distance(self, p1, p2, img=None, draw=True, r=15, t=3):
        """Calculate distance between two landmarks and optionally draw."""
        x1, y1 = p1
        x2, y2 = p2
        
        # Calculate distance
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        
        # Draw if image is provided
        if img is not None and draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        
        return length, [x1, y1, x2, y2, cx, cy]