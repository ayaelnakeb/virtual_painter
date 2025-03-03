import cv2
import numpy as np
import os
import time
import math

from hand_tracking import HandDetector
from gesture_detector import GestureDetector

def create_header_images():
    """Create header images if they don't exist."""
    os.makedirs('header/colors', exist_ok=True)
    
    # Create shape icons
    shapes = {
        'circle': (255, 255, 255),
        'rectangle': (255, 255, 255),
        'freehand': (255, 255, 255),
        'eraser': (255, 255, 255),
        'text': (255, 255, 255)
    }
    
    for shape, color in shapes.items():
        img = np.zeros((100, 100, 3), np.uint8)
        
        if shape == 'circle':
            cv2.circle(img, (50, 50), 40, color, 2)
        elif shape == 'rectangle':
            cv2.rectangle(img, (10, 10), (90, 90), color, 2)
        elif shape == 'freehand':
            points = [(20, 20), (40, 10), (60, 30), (80, 60), (60, 80), (30, 70)]
            for i in range(len(points)-1):
                cv2.line(img, points[i], points[i+1], color, 2)
        elif shape == 'eraser':
            cv2.rectangle(img, (20, 30), (80, 70), color, -1)
            cv2.putText(img, "ERASE", (25, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        elif shape == 'text':
            cv2.putText(img, "T", (25, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 3)
        
        cv2.imwrite(f'header/{shape}.png', img)
    
    # Create color icons
    colors = {
        'red': (0, 0, 255),
        'blue': (255, 0, 0),
        'green': (0, 255, 0),
        'yellow': (0, 255, 255),
        'purple': (255, 0, 255)
    }
    
    for color_name, bgr_color in colors.items():
        img = np.zeros((100, 100, 3), np.uint8)
        cv2.circle(img, (50, 50), 40, bgr_color, -1)
        cv2.imwrite(f'header/colors/{color_name}.png', img)

def main():
    # --- Setup & Init ---
    draw_color = (0, 0, 255)  # Default color: Red
    brush_thickness = 15
    eraser_thickness = 50
    text_to_draw = "HELLO"    # Default text for "text" mode

    create_header_images()  # create images if not present

    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    detector = HandDetector(detection_confidence=0.8, tracking_confidence=0.8)
    gesture_detector = GestureDetector()
    
    canvas = np.zeros((720, 1280, 3), np.uint8)

    # Load shape images
    shape_images = {}
    for shape in ['circle', 'rectangle', 'freehand', 'eraser', 'text']:
        path = f'header/{shape}.png'
        if os.path.exists(path):
            shape_images[shape] = cv2.imread(path)
        else:
            shape_images[shape] = np.zeros((100, 100, 3), np.uint8)

    # Load color images
    color_images = {}
    for color in ['red', 'blue', 'green', 'yellow', 'purple']:
        path = f'header/colors/{color}.png'
        if os.path.exists(path):
            color_images[color] = cv2.imread(path)
        else:
            color_images[color] = np.zeros((100, 100, 3), np.uint8)
    
    # Header Layout
    header_height = 100
    shape_x_positions = [i * 100 for i in range(5)]  # 5 shapes
    color_x_positions = [i * 100 for i in range(5)]  # 5 colors

    current_mode = 'freehand'
    px, py = 0, 0
    previous_time = 0

    # --- Selection "Tap" Logic ---
    selection_hold_frames = 0            # how many consecutive frames we've seen a "tap"
    selection_hold_threshold = 5         # frames required for a tap to count

    while True:
        success, img = cap.read()
        if not success:
            break
        img = cv2.flip(img, 1)

        # Detect hands
        img = detector.find_hands(img)
        landmark_list = detector.find_position(img)

        # Determine finger states
        fingers = [0,0,0,0,0]
        if landmark_list:
            fingers = detector.fingers_up(landmark_list)

        # Draw the header background
        img[0:header_height, 0:1280] = (50, 50, 50)

        # Draw shape icons
        for i, shape_name in enumerate(shape_images.keys()):
            x0 = shape_x_positions[i]
            img[0:100, x0:x0+100] = shape_images[shape_name]
            # highlight if selected
            if current_mode == shape_name:
                cv2.rectangle(img, (x0, 0), (x0+100, 100), (0, 255, 0), 3)
        
        # Draw color icons
        for i, color_name in enumerate(color_images.keys()):
            x_pos = shape_x_positions[-1] + 100 + color_x_positions[i]
            img[0:100, x_pos:x_pos+100] = color_images[color_name]
            if (
                (color_name == 'red'     and draw_color == (0, 0, 255)) or
                (color_name == 'blue'    and draw_color == (255, 0, 0)) or
                (color_name == 'green'   and draw_color == (0, 255, 0)) or
                (color_name == 'yellow'  and draw_color == (0, 255, 255)) or
                (color_name == 'purple'  and draw_color == (255, 0, 255))
            ):
                cv2.rectangle(img, (x_pos, 0), (x_pos+100, 100), (0, 255, 0), 3)

        # ----------------------------------------------------------------------
        # EASIER GESTURES:
        # 1) If index+middle up => erasing
        # 2) If index only => either drawing OR selecting in header
        # ----------------------------------------------------------------------

        if landmark_list:
            x1, y1 = landmark_list[8][1], landmark_list[8][2]  # index fingertip

            # === ERASER MODE: index & middle up ===
            if gesture_detector.detect_erasing_gesture(fingers):
                if current_mode != 'eraser':
                    current_mode = 'eraser'
                    px, py = 0, 0
                    print("Switched to eraser mode")
                
                if y1 > header_height:
                    if px == 0 and py == 0:
                        px, py = x1, y1
                    else:
                        cv2.line(canvas, (px, py), (x1, y1), (0,0,0), eraser_thickness)
                    px, py = x1, y1

            # === INDEX ONLY (no pinch needed) ===
            elif gesture_detector.detect_drawing_gesture(fingers):
                # If the index fingertip is in the header region => "tap" for selection
                if y1 < header_height:
                    # We'll accumulate frames
                    selection_hold_frames += 1
                    # Once we hold for 'N' consecutive frames, we do a selection
                    if selection_hold_frames > selection_hold_threshold:
                        selection_hold_frames = 0  # reset
                        # figure out if we tapped a shape or color
                        # check shapes
                        for i, shape_name in enumerate(shape_images.keys()):
                            x_start = shape_x_positions[i]
                            if x_start < x1 < x_start + 100:
                                current_mode = shape_name
                                print(f"Selected mode: {current_mode}")
                                px, py = 0, 0
                        
                        # check colors
                        for i, color_name in enumerate(color_images.keys()):
                            x_pos = shape_x_positions[-1] + 100 + color_x_positions[i]
                            if x_pos < x1 < x_pos + 100:
                                if color_name == 'red':
                                    draw_color = (0,0,255)
                                elif color_name == 'blue':
                                    draw_color = (255,0,0)
                                elif color_name == 'green':
                                    draw_color = (0,255,0)
                                elif color_name == 'yellow':
                                    draw_color = (0,255,255)
                                elif color_name == 'purple':
                                    draw_color = (255,0,255)
                                print(f"Selected color: {color_name}")
                else:
                    # Not in the header => freehand drawing or shape preview
                    selection_hold_frames = 0  # reset since we're out of header region

                    # If we were erasing, switch back
                    if current_mode == 'eraser':
                        current_mode = 'freehand'
                        px, py = 0, 0
                        print("Switched to freehand mode")

                    # FREEHAND
                    if current_mode == 'freehand':
                        if px == 0 and py == 0:
                            px, py = x1, y1
                        else:
                            cv2.line(canvas, (px, py), (x1, y1), draw_color, brush_thickness)
                        px, py = x1, y1
                    
                    # SHAPES / TEXT
                    elif current_mode in ['circle','rectangle','text']:
                        # If we haven't set a start point:
                        if px == 0 and py == 0:
                            px, py = x1, y1
                        else:
                            # If circle/rectangle, show a preview while index is down
                            if current_mode in ['circle','rectangle']:
                                preview = img.copy()
                                if current_mode == 'circle':
                                    radius = int(math.hypot(x1 - px, y1 - py))
                                    cv2.circle(preview, (px, py), radius, draw_color, brush_thickness)
                                else:  # rectangle
                                    cv2.rectangle(preview, (px, py), (x1, y1), draw_color, brush_thickness)
                                img = preview
                            # For text, you might directly place text on pinch or on release
                            # But for now, let's keep text placement on "pinch" if you want.
                
            else:
                # No recognized gesture
                px, py = 0, 0
                selection_hold_frames = 0  # reset

        # Merge the canvas with the webcam image
        img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, img_inv = cv2.threshold(img_gray, 1, 255, cv2.THRESH_BINARY_INV)
        img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, img_inv)
        img = cv2.bitwise_or(img, canvas)

        # Debug circle on index finger
        if landmark_list and len(landmark_list) > 8:
            cv2.circle(img, (landmark_list[8][1], landmark_list[8][2]), 10, (0,255,0), cv2.FILLED)

        # Show FPS
        current_time = time.time()
        fps = 1 / (current_time - previous_time + 1e-8)
        previous_time = current_time
        cv2.putText(img, f'FPS: {int(fps)}', (10,700), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        # Show current mode
        cv2.putText(img, f'Mode: {current_mode}', (1000,700), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        cv2.imshow("Virtual Painter", img)
        
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break
        elif key == ord('c'):
            canvas = np.zeros((720,1280,3), np.uint8)
            print("Canvas cleared")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
