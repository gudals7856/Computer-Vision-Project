import cv2
import numpy as np
import math

img1_origin = cv2.imread('frame1.jpg')
img2_origin = cv2.imread('frame2.jpg')

img1 = cv2.resize(img1_origin, dsize=(1280, 720), interpolation=cv2.INTER_AREA)
img2 = cv2.resize(img2_origin, dsize=(1280, 720), interpolation=cv2.INTER_AREA)

height, width, _ = img1.shape #720 1280

# 그레이스케일로 변환
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

#lukas-kanade 알고리즘 적용
def lukasKanade(y, x):

    A = np.zeros((9, 2), np.uint8)
    b = np.zeros((9, 1), np.uint8)

    # 자신과 주위8개 픽셀에 대해 모두 수행
    list_tmp = [-1, 0, 1]
    num = 0
    for m in list_tmp:
        for n in list_tmp:
            tmpy = y + m
            tmpx = x + n

            ay = int(gray1[tmpy, tmpx+1]) - int(gray1[tmpy, tmpx])
            A[num][0] = ay
            ax = int(gray1[tmpy+1, tmpx]) - int(gray1[tmpy, tmpx])
            A[num][1] = ax
            at = int(gray2[tmpy, tmpx]) - int(gray1[tmpy, tmpx])
            b[num][0] = at

            num = num+1


    # Normal Equation 해결
    tmp = np.dot(A.T, A)

    # 역행렬이 존재하지 않는 경우
    if np.linalg.det(tmp) == 0:
        return x, y

    inverse_arr = np.linalg.inv(tmp)
    vt = np.dot(np.dot(inverse_arr, A.T), b)

    print(vt)

    v = int(vt[0])
    u = int(vt[1])

    return v, u

# pt1과 pt2를 화면에 표시
for y in range(10,height,10):
    for x in range(10,width,10):
        v, u = lukasKanade(y, x)
        cv2.circle(gray1, (x, y), 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.arrowedLine(gray1, (x, y), (x + u, y + v), (255, 255, 255), 1)
    else:
        continue

cv2.imshow('result', gray1)
cv2.waitKey()
cv2.destroyAllWindows()

