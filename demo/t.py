import cv2
print("hello world")
im = cv2.imread('/home/jeff/Downloads/IMG_0942.jpg')
cv2.imshow(winname="Face", mat=im)
# cv2.namedWindow("hello world")
# cv2.namedWindow("Posture Analyzer", flags=(cv2.WINDOW_GUI_NORMAL + cv2.WINDOW_AUTOSIZE))
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print("goodbye world")