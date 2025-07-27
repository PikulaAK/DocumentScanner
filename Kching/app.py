import cv2
import numpy as np
from gpiozero import Button
from picamera2 import Picamera2, Preview
import time

button = Button(7)

picam2 = Picamera2()

preview_camera_config = picam2.create_preview_configuration()
still_camera_config = picam2.create_still_configuration()

picam2.configure(preview_camera_config)
picam2.start_preview(Preview.DRM)

picam2.start()

button.wait_for_press()

picam2.stop()
picam2.configure(still_camera_config)
picam2.start()

picam2.capture_file("test.jpg")

image = cv2.imread("test.jpg")
original = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 1: Apply Thresholding to create a binary image
_, thresh = cv2.threshold(gray, 150, 500, cv2.THRESH_BINARY_INV)  # Adjust the threshold value as needed

# Step 2: Use Morphological Operations to Close Gaps
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

edged = cv2.Canny(closed, 100, 200)

# Find the contours in the edged image, keeping only the largest ones
contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True) 
largest_contour = contours[0]

epsilon = 0.02 * cv2.arcLength(largest_contour, True) 
approx = cv2.approxPolyDP(largest_contour, epsilon, True)

if len(approx) == 4: 
    pts = approx.reshape(4, 2) 
else: 
    raise ValueError("Couldn't find a rectangular contour")

rect = np.zeros((4, 2), dtype="float32")
s = pts.sum(axis=1)
rect[0] = pts[np.argmin(s)]
rect[2] = pts[np.argmax(s)]

diff = np.diff(pts, axis=1)
rect[1] = pts[np.argmin(diff)]
rect[3] = pts[np.argmax(diff)]

width_a = np.sqrt(((rect[2][0] - rect[3][0]) ** 2) + ((rect[2][1] - rect[3][1]) ** 2))
width_b = np.sqrt(((rect[1][0] - rect[0][0]) ** 2) + ((rect[1][1] - rect[0][1]) ** 2))
max_width = max(int(width_a), int(width_b))

height_a = np.sqrt(((rect[1][0] - rect[2][0]) ** 2) + ((rect[1][1] - rect[2][1]) ** 2)) 
height_b = np.sqrt(((rect[0][0] - rect[3][0]) ** 2) + ((rect[0][1] - rect[3][1]) ** 2))
max_height = max(int(height_a), int(height_b))

dst = np.array([
    [0,0],
    [max_width - 1, 0],
    [max_width - 1, max_height - 1],
    [0, max_height - 1]
], dtype="float32")

M = cv2.getPerspectiveTransform(rect, dst) 
warped = cv2.warpPerspective(image, M, (max_width, max_height))

# Convert the final warped image to grayscale
warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

def BrightnessContrast(brightness=0): 
      
    effect = controller(warped_gray, 
                        265, 
                        155)
    # cv2.namedWindow("effect", cv2.WINDOW_NORMAL)
    # cv2.setWindowProperty("effect", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FREERATIO)
    # cv2.imshow('effect', effect)
    cv2.imwrite("C:\Data\Projects\Python\Kching\scannedDocuments"+ "\\test.jpg", effect) 

def controller(img, brightness=255, contrast=127): 
    brightness = int((brightness - 0) * (255 - (-255)) / (510 - 0) + (-255)) 
  
    contrast = int((contrast - 0) * (127 - (-127)) / (254 - 0) + (-127)) 
  
    if brightness != 0: 
  
        if brightness > 0: 
  
            shadow = brightness 
  
            max = 255
  
        else: 
  
            shadow = 0
            max = 255 + brightness 
  
        al_pha = (max - shadow) / 255
        ga_mma = shadow 
  
        # The function addWeighted  
        # calculates the weighted sum  
        # of two arrays 
        cal = cv2.addWeighted(img, al_pha, 
                              img, 0, ga_mma) 
  
    else: 
        cal = img 
  
    if contrast != 0: 
        Alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast)) 
        Gamma = 127 * (1 - Alpha) 
  
        # The function addWeighted calculates 
        # the weighted sum of two arrays 
        cal = cv2.addWeighted(cal, Alpha, 
                              cal, 0, Gamma) 
  
    # putText renders the specified 
    # text string in the image. 
    cv2.putText(cal, 'B:{},C:{}'.format(brightness,  
                                        contrast),  
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,  
                1, (0, 0, 255), 2) 
  
    return cal 

#Grayscale window

BrightnessContrast(0) 


cv2.waitKey(0)
cv2.destroyAllWindows()