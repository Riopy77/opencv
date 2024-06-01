from encodings import oem

import cv2
import imutils.contours
import myutils
import numpy as np
import pytesseract
from imutils import contours




def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img = cv2.imread('number.png')
cv2.imshow("img",img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ref, thre = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY_INV)

refCnts, hierarchy = cv2.findContours(thre.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, refCnts, -1, (0, 0, 255), 3)
refCnts = imutils.contours.sort_contours(refCnts, "left-to-right")[0]
digits = {}


for (i, c) in enumerate(refCnts):
    (x, y, w, h) = cv2.boundingRect(c)
    roi = thre[y:y + h, x:x + w]
    roi = cv2.resize(roi, (30, 50))
    digits[i] = roi

print(len(digits))


def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0

    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # 计算外接矩形 boundingBoxes是一个元组
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]  # 用一个最小的矩形，把找到的形状包起来x,y,h,w
    # sorted排序
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))

    return cnts, boundingBoxes  # 轮廓和boundingBoxess

ttttu = cv2.imread("card.png")

hui1 = cv2.cvtColor(ttttu, cv2.COLOR_BGR2GRAY)

tophat = cv2.morphologyEx(hui1, cv2.MORPH_TOPHAT, np.ones((2, 2), np.uint8))

gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)

gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
gradX = gradX.astype("uint8")

gradX = cv2.morphologyEx(gradX, cv2.MORPH_GRADIENT, np.ones((2, 4), np.uint8))

thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

thresh1 = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((5, 8), np.uint8))

thresh = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, np.ones((14, 25), np.uint8))

# 计算轮廓
threshCnts, his = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
cnts = threshCnts
cur_img = ttttu.copy()
cv2.drawContours(cur_img, cnts, -1, (0, 0, 255), 2)
cv2.imshow('contour_card', cur_img)

n = len(cnts)
for i in range(n):
    x, y, w, h = cv2.boundingRect(cnts[i])
    print("x:", x, "y:", y, "w:", w, "h:", h)

locs = []

for (i, c) in enumerate(cnts):
    # 计算矩形
    (x, y, w, h) = cv2.boundingRect(c)
    if (80 < w < 100) and (25 < h < 35):
            locs.append((x, y, w, h))

locs = sorted(locs, key=lambda x: x[0])
print(locs)

output = []
for (i, (gx, gy, gw, gh)) in enumerate(locs):
    groupOutput = []
    group = hui1[gy - 4:gy + gh + 4, gx - 4:gx + gw + 4]  # 获取轮廓及其周围数据,加五减五的作用是将获取到的坐标位上下左右偏移一点，方便匹配
    cv_show('group', group)
    group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv_show('group_' + str(i), group)
    digitCnts = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    # 排序
    digitCnts = imutils.contours.sort_contours(digitCnts, "left-to-right")[0]

    for c in digitCnts:
        z = 0
        (x, y, w, h) = cv2.boundingRect(c)
        roi = group[y:y + h, x:x + w]
        roi = cv2.resize(roi, (40, 50))
        cv_show("roi_"+str(z),roi)













