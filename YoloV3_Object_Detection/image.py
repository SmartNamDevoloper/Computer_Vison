import numpy as np
import cv2
img = cv2.imread("C:\\Users\\Administrator\\Documents\\opencv\\yolo\\bus.jpg")
cv2.imshow("Image",img)
cv2.imshow("Blue",img[:,:,0])
cv2.imshow("Green",img[:,:,1])
cv2.imshow("Red",img[:,:,2])
cv2.waitKey(0)
cv2.destroyAllWindows()
