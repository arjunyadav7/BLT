import mediapipe as mp
import cv2 as cv

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

cam = cv.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while True:
        ret, frame = cam.read()

        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        #making detections
        results = holistic.process(image)
        # print(results.face_landmarks)

        # re-colouring image to render using opencv
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

        #Drawing 1. Face Landmarks
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)

        #Drawing 2. Left Hand Landmarks
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        #Drawing 3. Right Hand Landmarks
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        #Drawing 4. Body Pose Landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        
        cv.imshow("holistic MediaPipe", image)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

cam.release()
cv.destroyAllWindows() 
