import mediapipe as mp
import numpy as np
import cv2

from utils.track_hands import render_landmarks
from utils.HandTracking import HandDetector

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

def open_camera(video_device = 0):
    cap = cv2.VideoCapture(video_device)

    # Instance Mediapipe
    # Pose accessing Pose estimation model
    with mp_pose.Pose(min_detection_confidence = 0.5,
                    min_tracking_confidence = 0.5) as pose:
        
        while cap.isOpened():
            ret, frame = cap. read()
            if not ret:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # Detect & render
            
            # Recolor to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Make detection
            results = pose.process(image)
            # results = pose.process(image)
            
            # Recolor to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)



            # Extrack Landmarks
            try:
                landmarks = results.pose_landmarks.landmark
            except:
                pass

            # Render detections
            render_landmarks(image, results)
        






            cv2.imshow('Mediapipe feed', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

# open_camera(0)


    
# mp_drawing = mp.solutions.drawing_utils
# mp_pose = mp.solutions.pose
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands()

def open_cam2(video_device = 0):

    cap = cv2.VideoCapture(video_device)

    # Instance Mediapipe
    # Hand accessing Hand estimation model
    hand = HandDetector()

    with hands:
        while cap.isOpened():
            ret, frame = cap. read()
            if not ret:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # Detect & render
            
            # Recolor to RGB
            # image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # image.flags.writeable = False


            img = hand.findHands(frame)
            # lmList = hand.findPosition(img, draw=False)


            
            cv2.imshow('Mediapipe feed', frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()



open_cam2(0)