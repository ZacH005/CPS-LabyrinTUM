#!/usr/bin/env python3
import cv2

# Generate a single large ChArUco board for printing
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_100)

# (cols, rows), larger squares for better distance detection
board = cv2.aruco.CharucoBoard((5, 5), squareLength=60, markerLength=45, dictionary=aruco_dict)

# Generate at high resolution for printing (2000x2000 px)
img = board.generateImage((2000, 2000))
cv2.imwrite("calibration/charuco_board.png", img)
print("Generated charuco_board.png - ready to print")