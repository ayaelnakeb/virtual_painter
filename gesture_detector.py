import cv2
import time
import math

class GestureDetector:
    def __init__(self):
        self.previous_time = 0
    
    def detect_drawing_gesture(self, fingers):
        """
        Detect if the hand is in drawing gesture
        Drawing gesture: Only index finger is up
        """
        return fingers == [0, 1, 0, 0, 0]
    
    def detect_erasing_gesture(self, fingers):
        """
        Detect if the hand is in erasing gesture
        Erasing gesture: Index and middle fingers are up
        """
        return fingers == [0, 1, 1, 0, 0]
    
    def detect_selection_gesture(self, fingers):
        """
        Detect if the hand is in selection gesture
        Selection gesture: All fingers are up
        """
        return fingers == [1, 1, 1, 1, 1]
    
    def check_click(self, landmark_list, detector):
        """
        Check if a click gesture is performed
        Click gesture: Index and thumb tips are close together
        """
        if not landmark_list or len(landmark_list) < 21:
            return False
        
        # Get positions of thumb and index finger tips
        thumb_tip = (landmark_list[4][1], landmark_list[4][2])
        index_tip = (landmark_list[8][1], landmark_list[8][2])
        
        # Calculate distance
        distance, _ = detector.find_distance(thumb_tip, index_tip)
        
        # If distance is less than threshold, consider it a click
        return distance < 40