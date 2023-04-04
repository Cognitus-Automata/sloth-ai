import cv2
import time
import mediapipe as mp
import math
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from Detector import track_hands
from pynput.mouse import Button, Controller

# Create an instance of mouse Controller
mouse = Controller()

cap = cv2.VideoCapture(0)
ptime = 0
isClicked = False
isGrabed = False
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles= mp.solutions.drawing_styles

with mp_hands.Hands(static_image_mode = False,
                    max_num_hands = 1, 
                    model_complexity = 1,
                    min_detection_confidence = 0.7) as hands:
    while True:

        ret, image = cap.read()
        if not ret:
            break
        image = cv2.flip(image, 1)
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ctime = time.time()
        fps = int(1/(ctime-ptime))
        results = hands.process(image)
        # lmList = hands.findPosition(img, draw=False)
            # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_height, image_width, _ = image.shape
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Render detections

                # print('hand_landmarks:', hand_landmarks)
                # print(
                #     f'Index finger tip coordinates: (',
                #     f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                #     f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
                # )
                xIndex = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
                yIndex = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y

                xThumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x
                yThumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y

                xMiddle = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x
                yMiddle = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y

                xWrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
                yWrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y

                # print(xIndex*image_width,yThumb*image_height)
                mouse.position = (round(xWrist*1920,1),round(yWrist*1080,1))

                print(round(xIndex*1920,1),round(yThumb*1080,1))

                dx, dy = xIndex - xThumb , yIndex - yThumb
                dx3, dy3 = xMiddle - xThumb, yMiddle - yThumb

                dist = np.sqrt(np.sum((dx** 2 , dy** 2)))  
                # dist3 = np.sqrt(np.sum(math.pow(Middle,2)  , Thumb** 2))
                dist3 = np.sqrt(np.sum((dx3** 2 , dy3** 2)))  

                if dist <= 0.02:
                    isClicked = True
                    mouse.press(Button.left)
                    print(isClicked)
                elif dist <= 0.02 and dist3 <= 0.02:
                    isGrabed = True
                    mouse.press(Button.left)
                    print(isClicked)
                else:
                    isClicked = False
                    mouse.release(Button.left)
                    print(isClicked)
                mp_drawing.draw_landmarks(
                                        image, 
                                        hand_landmarks,
                                        # mp_pose.POSE_CONNECTIONS,
                                        mp_hands.HAND_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(36, 182, 255), thickness=1, circle_radius=2),
                                        mp_drawing.DrawingSpec(color=(10,226,130), thickness=1, circle_radius=2)
                                        )
                # track_hands.render_landmarks(image, results)


        font = cv2.FONT_HERSHEY_DUPLEX

        cv2.putText(image,
                str(fps),
                (50, 50),
                font, 1,
                (220, 220, 220),
                2,
                cv2.LINE_4)
        ptime = ctime
        cv2.imshow("Frames", image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


