import pickle
from typing import Any

import mediapipe as mp
import os
import cv2
import matplotlib.pyplot as plt

DATA_DIR = './data'
labels = []
data: list[Any] = []
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True,min_detection_confidence=0.3)

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):

        img = cv2.imread(os.path.join(DATA_DIR, dir_,img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:

            for hand_landmark in results.multi_hand_landmarks:
                data_aux = []
                for i in range(len(hand_landmark.landmark)):
                    x=hand_landmark.landmark[i].x
                    y=hand_landmark.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)
                if data_aux:
                    data.append(data_aux)
                    labels.append(dir_)

f = open('data.pickle', 'wb')
pickle.dump({'data':data, 'labels':labels},f)
f.close()