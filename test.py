import numpy as np
import cv2
import imutils
import pytesseract
import pandas as pd
import time

image = cv2.imread('np.jpeg')
image = imutils.resize(image, width=500)

cv2.imshow("Original Image", image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 11, 17, 17)

edged = cv2.Canny(gray, 170, 200)

cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = [cnt.astype(np.float32) for cnt in cnts]
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]

NumberPlateCnt = None

for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
        NumberPlateCnt = approx
        break

if NumberPlateCnt is not None and len(NumberPlateCnt) > 0:
    mask = np.zeros(gray.shape, np.uint8)
    NumberPlateCnt = NumberPlateCnt.astype(int)
    cv2.fillPoly(mask, [NumberPlateCnt], 255)  # Fill contour onto mask

    if np.count_nonzero(mask) > 0:
        new_image = cv2.bitwise_and(image, image, mask=mask)
        cv2.imshow("Final Image", new_image)
        config = ('-l eng --oem 1 --psm 3')
        text = pytesseract.image_to_string(new_image, config=config)
        raw_data = {'date': [time.asctime(time.localtime(time.time()))],
                    'v_number': [text]}
        df = pd.DataFrame(raw_data, columns=['date', 'v_number'])
        df.to_csv('data.csv')
        print("Detected number plate:", text)
    else:
        print("Error: Mask is empty.")
else:
    print("Number plate contour not found.")

cv2.waitKey(0)
cv2.imwrite("output.jpeg", new_image)
cv2.destroyAllWindows()