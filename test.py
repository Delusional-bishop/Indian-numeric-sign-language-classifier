import pickle
# from curses.textpad import rectangle

import cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(0)

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True,min_detection_confidence=0.3)
labels_dict = {0:'1',1:'2',2:'3',3:'4',4:'5',5:'6',6:'7',7:'8',8:'9',9:'0'}
while True:
    ret, frame = cap.read()
    H,W,_ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
            frame,
            hand_landmark,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmark in results.multi_hand_landmarks:
            data_aux = []
            x_=[]
            y_=[]
            for i in range(len(hand_landmark.landmark)):
                x = hand_landmark.landmark[i].x
                y = hand_landmark.landmark[i].y
                data_aux.append(x)
                data_aux.append(y)
                x_.append(x)
                y_.append(y)

        x1 = int(min(x_)*W)
        y1 = int(min(y_)*H)
        x2 = int(max(x_)*W)
        y2 = int(max(y_)*H)

        prediction = model.predict([np.asarray(data_aux)])
        predictcted_character = labels_dict[int(prediction[0])]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 1)
        cv2.putText(frame, predictcted_character, (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 2,
                    cv2.LINE_AA)
    cv2.imshow('frame', frame)
    cv2.waitKey(1)


cap.release()
cv2.destroyAllWindows()