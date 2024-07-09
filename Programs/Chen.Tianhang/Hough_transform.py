import cv2
import numpy as np
from matplotlib import pyplot as plt

def nothing(x):  # 滑动条的回调函数
    pass

src = cv2.imread('D://Desktop/20210112_030017_1suff (2).jpg')
srcBlur = cv2.GaussianBlur(src, (3, 3), 0)
gray = cv2.cvtColor(srcBlur, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)
WindowName = 'Approx'  # 窗口名
cv2.namedWindow(WindowName, cv2.WINDOW_AUTOSIZE)  # 建立空窗口

cv2.createTrackbar('threshold', WindowName, 0, 60, nothing)  # 创建滑动条

while(1):
    img = src.copy()
    threshold = 100 + 2 * cv2.getTrackbarPos('threshold', WindowName)  # 获取滑动条值

    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold)
    for line in lines:
        rho = line[0][0]
        theta = line[0][1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imshow(WindowName, img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()

# import cv2
# import numpy as np
#
# img = cv2.imread('D:/Desktop/20210112_030017_1suff.png')
# img1 = img.copy()
# img2 = img.copy()
# img = cv2.GaussianBlur(img, (3, 3), 0)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# edges = cv2.Canny(gray, 50, 150, apertureSize=3)
# lines = cv2.HoughLines(edges, 1, np.pi / 180, 110)
#
# for line in lines:
#     rho = line[0][0]
#     theta = line[0][1]
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a * rho
#     y0 = b * rho
#     x1 = int(x0 + 1000 * (-b))
#     y1 = int(y0 + 1000 * (a))
#     x2 = int(x0 - 1000 * (-b))
#     y2 = int(y0 - 1000 * (a))
#
#     cv2.line(img1, (x1, y1), (x2, y2), (0, 0, 255), 2)
#
# lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 30, 300, 5)
#
# for line in lines:
#     x1 = line[0][0]
#     y1 = line[0][1]
#     x2 = line[0][2]
#     y2 = line[0][3]
#     cv2.line(img2, (x1, y1), (x2, y2), (0, 255, 0), 2)
#
# cv2.imshow('houghlines3', img1)
# cv2.imshow('edges', img2)
# cv2.waitKey(0)
# print(lines)

# ————————————————
# 版权声明：本文为CSDN博主「YukinoSiro」的原创文章，遵循CC
# 4.0
# BY - SA版权协议，转载请附上原文出处链接及本声明。
# 原文链接：https: // blog.csdn.net / yukinoai / article / details / 88366564
