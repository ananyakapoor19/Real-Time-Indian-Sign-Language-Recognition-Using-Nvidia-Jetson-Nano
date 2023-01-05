import cv2
import HandTrackingModule
import ClassificationModule
import numpy as np
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("keras_model.h5", "labels.txt")

offset = 50
imgSize = 300
counter = 0

labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M",
          "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"]

while True:
    success, img = cap.read()
    hands = detector.findHands(img, draw=False)

    filtered = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filtered = cv2.GaussianBlur(filtered, (5, 5), 2)
    # _, filtered = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    filtered = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    ret, filtered = cv2.threshold(filtered, 170, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # cv2.imshow("Original", img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize), np.uint8) * 255
        imgCrop = filtered[y - offset:y + h + offset, x - offset:x + w + offset]
        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        try:
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize

                gray2rgb = cv2.cvtColor(imgWhite, cv2.COLOR_GRAY2RGB)
                prediction, index = classifier.getPrediction(gray2rgb)
                print(labels[index])
                print(prediction)
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

                gray2rgb = cv2.cvtColor(imgWhite, cv2.COLOR_GRAY2RGB)
                prediction, index = classifier.getPrediction(gray2rgb)
                print(labels[index])  # if(
                print(prediction)

            if (x - offset > 0 and x + offset < img.shape[1] and y - offset > 0 and y + offset < img.shape[0]):
                cv2.imshow("Filtered", filtered)
                # cv2.imshow("Cropped", imgCrop)
                cv2.imshow("Final", imgWhite)

        except:
            print("Aghhh")

    key = cv2.waitKey(1)
    if key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
