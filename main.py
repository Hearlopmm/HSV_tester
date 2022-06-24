from __future__ import division

import cv2
import numpy as np
import struct


def cv_show(name, img):  # 定义一个函数，显示图片
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


pxl = {}
pyl = {}
t = 0
i = 0
rx = ry = gx = gy = bx1 = by1 = bx2 = by2 = bx3 = by3 = 0

cap = cv2.VideoCapture('D:/aaa/difcolour.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)  # 获取帧数
if cap.isOpened():
    ret, img = cap.read()
    while ret:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        lower_red1 = np.array([0, 80, 80])
        upper_red1 = np.array([7, 255, 255])
        mask = cv2.inRange(hsv, lower_red1, upper_red1)
        imgcanny = cv2.Canny(mask, 50, 100)  # canny轮廓检测
        _, fcon, hier = cv2.findContours(imgcanny, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)  # 寻找轮廓
        for c in fcon:  # 寻找每一个包围矩形
            x, y, w, h = cv2.boundingRect(c)
            if w < 50 or h < 40:
                continue
            cv2.rectangle(img, (x, y), (x + w, y + h), (80, 50, 180), 2)  # 画出所有矩形框
            cv2.putText(img, str('Red'), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 0, 220), 1)

        lower_red2 = np.array([140, 80, 50])
        upper_red2 = np.array([180, 255, 255])
        mask0 = cv2.inRange(hsv, lower_red2, upper_red2)
        # cv_show('?',mask0)
        imgcanny0 = cv2.Canny(mask0, 50, 100)  # canny轮廓检测
        _, fcon, hier = cv2.findContours(imgcanny0, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)  # 寻找轮廓
        for c in fcon:  # 寻找每一个包围矩形
            x, y, w, h = cv2.boundingRect(c)
            # print(w,h)
            if w < 50 or h < 40:
                continue
            cv2.rectangle(img, (x, y), (x + w, y + h), (80, 50, 180), 2)  # 画出所有矩形框
            cv2.putText(img, str('Red'), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 0, 220), 1)

        lower_green = np.array([35, 43, 43])
        upper_green = np.array([90, 255, 255])
        mask1 = cv2.inRange(hsv, lower_green, upper_green)
        # cv_show('?',mask1)
        imgcanny1 = cv2.Canny(mask1, 50, 100)  # canny轮廓检测
        # cv_show('canny', imgcanny1)
        _, fcon, hier = cv2.findContours(imgcanny1, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)  # 寻找轮廓
        for c in fcon:  # 寻找每一个包围矩形
            x, y, w, h = cv2.boundingRect(c)
            # print(w,h)
            if w < 30 or h < 80:
                continue
            # print(w, h)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 180, 0), 2)  # 画出所有矩形框
            cv2.putText(img, str('Green'), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 255, 0), 1)

        xl = 0
        yl = 0
        px = 0
        py = 0
        six = 6
        xll = [0, 0, 0]
        yll = [0, 0, 0]
        d = [0, 0, 0]
        pi = i
        i = 0
        lower_blue = np.array([90, 100, 34])
        upper_blue = np.array([118, 255, 255])
        mask2 = cv2.inRange(hsv, lower_blue, upper_blue)
        imgcanny2 = cv2.Canny(mask2, 50, 100)  # canny轮廓检测
        # cv_show('canny', imgcanny)
        _, fcon, hier = cv2.findContours(imgcanny2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)  # 寻找轮廓
        for c in fcon:  # 寻找每一个包围矩形
            x, y, w, h = cv2.boundingRect(c)
            if w < 20 or h < 20:
                continue
            if x == px and y == py:
                continue
            px = x
            py = y
            if t == 0:  # initial
                pxl[i] = (x + w) / 2
                pyl[i] = (y + h) / 2
                cv2.rectangle(img, (x, y), (x + w, y + h), (200, 0, 0), 2)  # 画出所有矩形框
                cv2.putText(img, str('Blue') + str(i), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 25, 20), 1)
            else:
                cv2.rectangle(img, (x, y), (x + w, y + h), (200, 0, 0), 2)  # 画出所有矩形框
                xl = (x + w) / 2
                yl = (y + h) / 2
                for m in range(3):
                    d[m] = np.square(abs(pxl[m] - xl)) + np.square(abs(pyl[m] - yl))
                n = d.index(min(d))
                six = six - n - 1
                xll[n] = xl
                yll[n] = yl
                cv2.putText(img, str('Blue') + str(n), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 25, 20), 1)
            i = i + 1
        if t != 0:
            if six > 0:
                xll[six - 1] = pxl[six - 1]
                yll[six - 1] = pyl[six - 1]
            pxl = xll
            pyl = yll
        t = 1
        print(i, pxl)
        cv2.waitKey(int(1000 / fps))  # 延时
        cv2.imshow('windows', img)
        ret, img = cap.read()

    cap.release()

'''
img = cv2.imread("8.jpg")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
'''
'''
lower_red1 = np.array([0, 100, 80])
upper_red1 = np.array([7, 255, 255])
mask = cv2.inRange(hsv, lower_red1, upper_red1)
imgcanny = cv2.Canny(mask, 50, 100)  # canny轮廓检测
_, fcon, hier = cv2.findContours(imgcanny, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)  # 寻找轮廓
for c in fcon:  # 寻找每一个包围矩形
    x, y, w, h = cv2.boundingRect(c)
    if w < 50 or h < 40:
        continue
    cv2.rectangle(img, (x, y), (x + w, y + h), (80, 50, 180), 2)  # 画出所有矩形框
    cv2.putText(img, str('Red'), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 0, 220), 1)

lower_red2 = np.array([156, 100, 50])
upper_red2 = np.array([180, 255, 255])
mask0 = cv2.inRange(hsv, lower_red2, upper_red2)
# cv_show('?',mask0)
imgcanny0 = cv2.Canny(mask0, 50, 100)  # canny轮廓检测
_, fcon, hier = cv2.findContours(imgcanny0, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)  # 寻找轮廓
for c in fcon:  # 寻找每一个包围矩形
    x, y, w, h = cv2.boundingRect(c)
    # print(w,h)
    if w < 50 or h < 40:
        continue
    cv2.rectangle(img, (x, y), (x + w, y + h), (80, 50, 180), 2)  # 画出所有矩形框
    cv2.putText(img, str('Red'), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 0, 220), 1)
'''
'''
lower_green = np.array([35, 43, 43])
upper_green = np.array([90, 255, 255])
mask1 = cv2.inRange(hsv, lower_green, upper_green)
# cv_show('?',mask1)
imgcanny1 = cv2.Canny(mask1, 50, 100)  # canny轮廓检测
# cv_show('canny', imgcanny1)
_, fcon, hier = cv2.findContours(imgcanny1, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)  # 寻找轮廓
for c in fcon:  # 寻找每一个包围矩形
    x, y, w, h = cv2.boundingRect(c)
    print(w,h)
    if w < 25 or h < 60:
        continue
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 180, 0), 2)  # 画出所有矩形框
    cv2.putText(img, str('Green'), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 255, 0), 1)


ij = 0
xl = 0
yl = 0
d = []

lower_blue = np.array([80, 100, 30])
upper_blue = np.array([124, 255, 255])
mask2 = cv2.inRange(hsv, lower_blue, upper_blue)
imgcanny2 = cv2.Canny(mask2, 50, 100)  # canny轮廓检测
# cv_show('canny', imgcanny2)
_, fcon, hier = cv2.findContours(imgcanny2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)  # 寻找轮廓
for c in fcon:  # 寻找每一个包围矩形
    x, y, w, h = cv2.boundingRect(c)
    if w < 30 or h < 30:
        continue
    cv2.rectangle(img, (x, y), (x + w, y + h), (200, 0, 0), 2)  # 画出所有矩形框
    # print(w, h)

cv_show('', img)
'''
