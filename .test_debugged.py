import numpy as np
import cv2
import imutils
import pytesseract
import pandas as pd
import time

# Load the image
image = cv2.imread('np.jpeg')

# Resize the image for easier processing
image = imutils.resize(image, width=500)

# Display the original image
cv2.imshow("Original Image", image)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply bilateral filter for noise reduction while keeping edges sharp
gray = cv2.bilateralFilter(gray, 11, 17, 17)

# Detect edges using Canny edge detection
edged = cv2.Canny(gray, 170, 200)

# Find contours
cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Ensure contours are in the correct datatype
cnts = [cnt.astype(np.float32) for cnt in cnts]

# Sort contours by area
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]

NumberPlateCnt = None

# Loop through contours to find the number plate contour
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
        NumberPlateCnt = approx
        break

# Check if the number plate contour is found
if NumberPlateCnt is not None and len(NumberPlateCnt) > 0:
    # Create a mask to isolate the number plate region
    mask = np.zeros(gray.shape, np.uint8)
    # Convert the points to integer datatype
    NumberPlateCnt = NumberPlateCnt.astype(int)
    cv2.fillPoly(mask, [NumberPlateCnt], 255)  # Fill contour onto mask

    # Check if the mask is not empty
    if np.count_nonzero(mask) > 0:
        new_image = cv2.bitwise_and(image, image, mask=mask)

        # Show the final image with the number plate region isolated
        cv2.imshow("Final Image", new_image)

        # Configuration for tesseract
        config = ('-l eng --oem 1 --psm 3')

        # Run OCR on the number plate region
        text = pytesseract.image_to_string(new_image, config=config)

        # Data to be stored in CSV
        raw_data = {'date': [time.asctime(time.localtime(time.time()))],
                    'v_number': [text]}

        # Create a DataFrame and store data in CSV
        df = pd.DataFrame(raw_data, columns=['date', 'v_number'])
        df.to_csv('data.csv')

        # Print recognized text
        print("Detected number plate:", text)
    else:
        print("Error: Mask is empty.")
else:
    print("Number plate contour not found.")

cv2.waitKey(0)
cv2.imwrite("output.jpeg", new_image)
cv2.destroyAllWindows()
