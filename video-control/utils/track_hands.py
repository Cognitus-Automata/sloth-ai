import mediapipe as mp
import numpy as np
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

def render_landmarks(image, results):
    mp_drawing.draw_landmarks(
            image, 
            results.pose_landmarks,
            # mp_pose.POSE_CONNECTIONS,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(36, 182, 255), thickness=1, circle_radius=2),
            mp_drawing.DrawingSpec(color=(10,226,130), thickness=1, circle_radius=2)
            )