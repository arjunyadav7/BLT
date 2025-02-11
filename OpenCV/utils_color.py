import numpy as np
import cv2 as cv

def get_limits(color):
    # Convert BGR color to a 1x1x3 NumPy array
    bgr_color = np.uint8([[color]])
    
    # Convert BGR to HSV
    hsv_color = cv.cvtColor(bgr_color, cv.COLOR_BGR2HSV)[0][0]
    hue = hsv_color[0]

    # Define the HSV range dynamically
    lower_hue = max(hue - 10, 0)  # Ensure hue is not negative
    upper_hue = min(hue + 10, 179)  # HSV hue max is 179
    
    lower_limit = np.array([lower_hue, 100, 100], dtype=np.uint8)  # S and V > 100 for good intensity
    upper_limit = np.array([upper_hue, 255, 255], dtype=np.uint8)

    return lower_limit, upper_limit
