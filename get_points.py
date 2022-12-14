import math

import cv2 as cv


def draw(img, start, end, color):
    thickness = 1
    line_type = 8
    cv.line(img, start, end, color, thickness, line_type)


def mousePoints(event, x, y, flags, params):
    if event == cv.EVENT_LBUTTONDOWN:
        print(x, y)


vid = cv.VideoCapture(".\\input\\1.mp4")
width = int(vid.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(vid.get(cv.CAP_PROP_FRAME_HEIGHT))
resolution = (width, height)

ret, frame = vid.read()
cv.imwrite('temp.jpg', frame)
vid.release()

img = cv.imread('temp.jpg')
resized_img = cv.resize(img, resolution, interpolation=cv.INTER_AREA)

for i in range(1, math.floor(height / 50) + 1):
    draw(resized_img, (0, 50 * i), (width, 50 * i), (0, 255, 0))
for i in range(1, math.floor(width / 50) + 1):
    draw(resized_img, (50 * i, 0), (50 * i, height), (0, 255, 0))

cv.imshow("Image", resized_img)
cv.setMouseCallback("Image", mousePoints)

cv.waitKey(0)
cv.destroyAllWindows()
