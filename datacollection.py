import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
counter = 0

folder = r"C:\Users\focus\Desktop\Sign-language detection\Data\cookie"

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        imgCrop = img[y - offset : y + h + offset, x - offset : x + w + offset]

        # Check if imgCrop is not empty
        if imgCrop.size > 0:
            imgCropShape = imgCrop.shape
            aspect_ratio = h / w

            if aspect_ratio > 1:
                k = imgSize / h
                w_cal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (w_cal, imgSize))

                # Check if imgResize is not empty
                if imgResize.size > 0:
                    imgResizeShape = imgResize.shape
                    w_gap = math.ceil((imgSize - w_cal) / 2)
                    imgWhite[:, w_gap : w_cal + w_gap] = imgResize
                else:
                    print("Error: Empty resized image!")
            else:
                k = imgSize / w
                h_cal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, h_cal))

                # Check if imgResize is not empty
                if imgResize.size > 0:
                    imgResizeShape = imgResize.shape
                    h_gap = math.ceil((imgSize - h_cal) / 2)
                    imgWhite[h_gap : h_cal + h_gap, :] = imgResize
                else:
                    print("Error: Empty resized image!")

            cv2.imshow('ImageCrop', imgCrop)
            cv2.imshow('ImageWhite', imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('s'):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)

