import cv2
import matplotlib.pyplot as plt

bottom_left=(0,0)
bottom_right=(0,600)
top_left=(600,0)
top_right=(600,600)

# images found in /calibration

def show_image(img, title):
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()

image = cv2.imread("calibration/CharUco_9.png")

show_image(image, "Board Calibration")
