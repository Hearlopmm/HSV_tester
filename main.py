# import the libraries
import cv2
import numpy as np
'''
# read the image
img = cv.imread("1.jpg")
# convert the BGR image to HSV colour space
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
# obtain the grayscale image of the original image
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# set the bounds for the red hue
lower_red = np.array([160, 100, 50])
upper_red = np.array([180, 255, 255])

# create a mask using the bounds set
mask = cv.inRange(hsv, lower_red, upper_red)


# create resizable windows for the images
cv.namedWindow("mask", cv.WINDOW_NORMAL)

# display the images
cv.imshow("mask", mask)

if cv.waitKey(0):
    cv.destroyAllWindows()
    '''

# 读取图像
img = cv2.imread("4.jpg")
# y1, y2, x1, x2
# img = img[0:579, 315:704]

# 转HSV
imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


# 回调函数必须要写
def empty(i):
    # 提取滑动条的数值 共6个
    hue_min = cv2.getTrackbarPos("Hue Min", "sb")
    hue_max = cv2.getTrackbarPos("Hue Max", "sb")
    sat_min = cv2.getTrackbarPos("Sat Min", "sb")
    sat_max = cv2.getTrackbarPos("Sat Max", "sb")
    val_min = cv2.getTrackbarPos("Val Min", "sb")
    val_max = cv2.getTrackbarPos("Val Max", "sb")

    # 颜色空间阈值
    lower = np.array([hue_min, sat_min, val_min])
    upper = np.array([hue_max, sat_max, val_max])
    # 根据颜色空间阈值生成掩膜
    imgMASK = cv2.inRange(imgHSV, lower, upper)
    cv2.imshow("Mask", imgMASK)


# 参数(‘窗口标题’,默认参数)
cv2.namedWindow("sb")
# 设置窗口大小
cv2.resizeWindow("sb", 640, 240)

# 第一个参数时滑动条的名字，
# 第二个参数是滑动条被放置的窗口的名字，
# 第三个参数是滑动条默认值，
# 第四个参数时滑动条的最大值，
# 第五个参数时回调函数，每次滑动都会调用回调函数。
cv2.createTrackbar("Hue Min", "sb", 140, 255, empty)
cv2.createTrackbar("Hue Max", "sb", 180, 255, empty)
cv2.createTrackbar("Sat Min", "sb", 80, 255, empty)
cv2.createTrackbar("Sat Max", "sb", 255, 255, empty)
cv2.createTrackbar("Val Min", "sb", 50, 255, empty)
cv2.createTrackbar("Val Max", "sb", 255, 255, empty)

# 调用函数
empty(0)

# 按任意键关闭
cv2.waitKey()