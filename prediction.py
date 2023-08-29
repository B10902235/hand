import cv2
import mediapipe as mp
from cvzone.HandTrackingModule import HandDetector
import numpy as np
from keras.models import load_model



def num2alphabet(text):    # 將預測結果轉為英文字母
    alphabet = 'ABCDEFGHIKLMNOPQRSTUVWXY'
    return alphabet[text]





def predict(img):
    detector = HandDetector(detectionCon=0.5, maxHands=1)  # cvzone,用於抓出手部位置

    model = load_model('signDot.h5')  # 關節點手勢辨識模型

    mp_hands = mp.solutions.hands  # mediapipe 偵測手掌方法

    hands_dot = mp_hands.Hands(
        model_complexity=1,  # 複雜度越高越準確，但會增加延遲
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    hands = detector.findHands(img, draw=False)  # 使用cvzone抓出手部位置
    if hands:  # 若畫面中有手，並成功偵測到
        hand1 = hands[0]
        x, y, w, h = hand1['bbox']  # 抓出手部座標
        # print(x, y, w, h, sep=' ')
        if w > h:  # 擷取手部的影像，並且確保擷取結果為爭方形，且不超出原圖範圍
            y = int(y - (w - h) / 2)
            h = w
        else:
            x = int(x - (h - w) / 2)
            w = h
        img = img[y - 20:y + h + 20, x - 20:x + w + 20]
        if x - 20 > 0 and x + w + 20 < 765 and y - 20 > 0:
            RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 轉RGB
            RGB = cv2.resize(RGB, (128, 128))  # 調整圖片大小
            results = hands_dot.process(RGB)  # 偵測手部關節點座標
            finger_points = []  # 記錄手指節點座標的陣列

            if results.multi_hand_landmarks:  # 若成功抓到座標
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in hand_landmarks.landmark:
                        # 將 21 個節點換算成座標，記錄到 finger_points
                        finger_points.append(i.x)
                        finger_points.append(i.y)
                finger_points = np.array(finger_points)  # 轉為numpy array
                finger_points = finger_points.reshape(1, 42)  # reshape
                # print(finger_points.shape)
                prediction = model.predict(finger_points, verbose=0)  # 輸出的是編碼
                index = np.argmax(prediction)  # 將編碼轉換後才是結果
                text = num2alphabet(index)
                return round(prediction.max(), 3), text
            else:
                print('mediapipe沒偵測到')
        else:
            print('手太靠近螢幕邊緣')
    else:
        print('no hands')

