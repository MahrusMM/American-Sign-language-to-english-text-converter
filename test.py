import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import math

class SignTranslatorApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        self.cap = cv2.VideoCapture(0)
        model_path = r"C:\Users\focus\Downloads\converted_keras\keras_model.h5"
        labels_path = r"C:\Users\focus\Downloads\converted_keras\labels.txt"
        self.detector = HandDetector(maxHands=1)
        self.classifier = Classifier(model_path, labels_path)  # Use the correct paths

        self.offset = 20
        self.imgSize = 300
        self.labels = ["Hello", "Thank you", "Yes"]

        self.canvas = tk.Canvas(window, width=self.imgSize, height=self.imgSize)
        self.canvas.pack()

        self.start_button = tk.Button(window, text="Start Translating", command=self.start_translating)
        self.start_button.pack()

        self.close_button = tk.Button(window, text="Close App", command=self.close_app)
        self.close_button.pack()

    def start_translating(self):
        while True:
            success, img = self.cap.read()
            imgOutput = img.copy()
            hands, img = self.detector.findHands(img)

            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']

                imgWhite = np.ones((self.imgSize, self.imgSize, 3), np.uint8) * 255
                imgCrop = img[y - self.offset:y + h + self.offset, x - self.offset:x + w + self.offset]

                # Check if imgCrop is not empty
                if not imgCrop.size:
                    print("Error: Empty imgCrop!")
                    continue  # Skip this iteration if imgCrop is empty

                imgCropShape = imgCrop.shape

                aspectRatio = h / w

                if aspectRatio > 1:
                    k = self.imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, self.imgSize))
                    
                    # Check if imgResize is not empty
                    if not imgResize.size:
                        print("Error: Empty imgResize!")
                        continue  # Skip this iteration if imgResize is empty
                    
                    imgResizeShape = imgResize.shape
                    wGap = math.ceil((self.imgSize - wCal) / 2)
                    imgWhite[:, wGap: wCal + wGap] = imgResize
                    prediction, index = self.classifier.getPrediction(imgWhite, draw=False)
                    print(f"Prediction: {prediction}, Index: {index}, Aspect Ratio > 1")

                else:
                    k = self.imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (self.imgSize, hCal))
                    
                    # Check if imgResize is not empty
                    if not imgResize.size:
                        print("Error: Empty imgResize!")
                        continue  # Skip this iteration if imgResize is empty
                    
                    imgResizeShape = imgResize.shape
                    hGap = math.ceil((self.imgSize - hCal) / 2)
                    imgWhite[hGap: hCal + hGap, :] = imgResize
                    prediction, index = self.classifier.getPrediction(imgWhite, draw=False)
                    print(f"Prediction: {prediction}, Index: {index}, Aspect Ratio <= 1")

                cv2.rectangle(imgOutput, (x - self.offset, y - self.offset - 70),
                              (x - self.offset + 400, y - self.offset + 60 - 50), (0, 255, 0), cv2.FILLED)

                cv2.putText(imgOutput, self.labels[index], (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
                cv2.rectangle(imgOutput, (x - self.offset, y - self.offset), (x + w + self.offset, y + h + self.offset),
                              (0, 255, 0), 4)

                cv2.imshow('ImageCrop', imgCrop)
                cv2.imshow('ImageWhite', imgWhite)

            cv2.imshow('Image', imgOutput)
            key = cv2.waitKey(1)
            if key == ord('q'):  # Press 'q' to exit the loop
                break

    def close_app(self):
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = SignTranslatorApp(root, "Sign Translator App")
    root.mainloop()
