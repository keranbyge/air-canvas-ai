import cv2
import mediapipe as mp
import numpy as np
from collections import deque

mpHands = mp.solutions.hands
hands = mpHands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

canvas = None

blue = (255,0,0)
green = (0,255,0)
red = (0,0,255)
yellow = (0,255,255)

drawColor = blue
brushThickness = 8
brushSize = "Medium"

xp, yp = 0, 0

tipIds = [4,8,12,16,20]

# Smoothing buffer
smoothing_buffer = deque(maxlen=5)

# Gesture stability buffer
gesture_buffer = deque(maxlen=5)

def count_fingers(lmList):
    """Improved finger counting with better logic"""
    fingers = []
    
    # Thumb (check x-axis for right hand)
    if lmList[tipIds[0]][1] < lmList[tipIds[0]-1][1]:
        fingers.append(1)
    else:
        fingers.append(0)
    
    # Other 4 fingers (check y-axis)
    for id in range(1, 5):
        if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
            fingers.append(1)
        else:
            fingers.append(0)
    
    return fingers

def smooth_position(x, y):
    """Smooth finger position to reduce jitter"""
    smoothing_buffer.append((x, y))
    if len(smoothing_buffer) > 0:
        avg_x = int(np.mean([pos[0] for pos in smoothing_buffer]))
        avg_y = int(np.mean([pos[1] for pos in smoothing_buffer]))
        return avg_x, avg_y
    return x, y

def get_stable_gesture(finger_count):
    """Buffer gestures to prevent flickering"""
    gesture_buffer.append(finger_count)
    if len(gesture_buffer) >= 3:
        return max(set(gesture_buffer), key=gesture_buffer.count)
    return finger_count

while True:
    success, frame = cap.read()
    
    if not success:
        break
    
    frame = cv2.flip(frame, 1)
    
    if canvas is None:
        canvas = np.zeros_like(frame)
    
    # Draw color palette
    cv2.rectangle(frame, (0,0), (160,65), blue, -1)
    cv2.rectangle(frame, (160,0), (320,65), green, -1)
    cv2.rectangle(frame, (320,0), (480,65), red, -1)
    cv2.rectangle(frame, (480,0), (640,65), yellow, -1)
    
    cv2.putText(frame, "BLUE", (40,45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(frame, "GREEN", (180,45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(frame, "RED", (360,45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(frame, "YELLOW", (500,45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
    
    # Convert to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Hand detection
    results = hands.process(rgb)
    
    lmList = []
    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
            
            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)
    
    # If hand detected
    if len(lmList) != 0:
        fingers = count_fingers(lmList)
        totalFingers = fingers.count(1)
        
        # Get stable gesture
        stableFingers = get_stable_gesture(totalFingers)
        
        # Index finger tip position
        x1, y1 = lmList[8][1:]
        
        # Smooth the position
        x1, y1 = smooth_position(x1, y1)
        
        # Brush size control - 3 fingers (Small)
        if stableFingers == 3 and fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1:
            brushThickness = 5
            brushSize = "Small"
            xp, yp = 0, 0
            cv2.putText(frame, "Brush: SMALL", (250, 250), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,0), 3)
        
        # Brush size control - 4 fingers (Medium)
        elif stableFingers == 4:
            brushThickness = 10
            brushSize = "Medium"
            xp, yp = 0, 0
            cv2.putText(frame, "Brush: MEDIUM", (220, 250), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,0), 3)
        
        # Color selection mode (index + middle finger up)
        elif stableFingers == 2 and fingers[1] == 1 and fingers[2] == 1:
            xp, yp = 0, 0
            cv2.circle(frame, (x1, y1), 15, (0,255,255), cv2.FILLED)
            
            if y1 < 65:
                if x1 < 160:
                    drawColor = blue
                    cv2.rectangle(frame, (0,0), (160,65), (255,255,255), 3)
                elif x1 < 320:
                    drawColor = green
                    cv2.rectangle(frame, (160,0), (320,65), (255,255,255), 3)
                elif x1 < 480:
                    drawColor = red
                    cv2.rectangle(frame, (320,0), (480,65), (255,255,255), 3)
                else:
                    drawColor = yellow
                    cv2.rectangle(frame, (480,0), (640,65), (255,255,255), 3)
        
        # Drawing mode (only index finger up)
        elif stableFingers == 1 and fingers[1] == 1:
            cv2.circle(frame, (x1, y1), brushThickness, drawColor, cv2.FILLED)
            
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            
            # Only draw if not in palette area
            if y1 > 65:
                cv2.line(canvas, (xp, yp), (x1, y1), drawColor, brushThickness)
            
            xp, yp = x1, y1
        
        # Clear screen (all 5 fingers up)
        elif stableFingers == 5:
            canvas = np.zeros_like(canvas)
            xp, yp = 0, 0
            cv2.putText(frame, "CLEARED!", (250, 250), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 5)
        
        else:
            xp, yp = 0, 0
    
    else:
        xp, yp = 0, 0
    
    # Merge canvas with frame
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, inv = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)
    inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
    
    frame = cv2.bitwise_and(frame, inv)
    frame = cv2.bitwise_or(frame, canvas)
    
    # Display brush size
    cv2.putText(frame, f"Brush Size: {brushSize}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
    
    # Instructions
    cv2.putText(frame, "1 Finger = Draw", (10, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    
    cv2.putText(frame, "2 Fingers = Color | 3 = Small | 4 = Medium", (10, 420),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    
    cv2.putText(frame, "5 Fingers = Clear", (10, 390),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    
    cv2.imshow("Air Canvas", frame)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
