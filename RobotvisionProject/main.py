import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread('frame1.jpg')
img2 = cv2.imread('frame2.jpg')

# img1 = cv2.resize(img1_origin, dsize=(1280, 720), interpolation=cv2.INTER_AREA)
# img2 = cv2.resize(img2_origin, dsize=(1280, 720), interpolation=cv2.INTER_AREA)

height, width, _ = img1.shape  # 720 1280 (y, x)

# 그레이스케일로 변환
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

dirX = [1, -1, 0]
dirY = [1, -1, 0]


# lukas-kanade 알고리즘 적용
def lukasKanade(x, y):  # 1280, 720
    A = np.zeros((9, 2), np.uint8)
    b = np.zeros((9, 1), np.uint8)

    # 자신과 주위8개 픽셀에 대해 모두 수행
    num = 0
    for m in dirX:
        for n in dirY:
            tmpy = y + n
            tmpx = x + m

            ay = int(gray1[tmpy + 1, tmpx]) - int(gray1[tmpy, tmpx])
            A[num][0] = ay
            ax = int(gray1[tmpy, tmpx + 1]) - int(gray1[tmpy, tmpx])
            A[num][1] = ax
            at = int(gray2[tmpy, tmpx]) - int(gray1[tmpy, tmpx])
            b[num][0] = -at

            num = num + 1

    # Normal Equation 해결
    tmp = np.dot(A.T, A)

    # 역행렬이 존재하지 않는 경우 예외. 자기 자신을 반환
    if np.linalg.det(tmp) == 0:
        # cv2.arrowedLine(img1, (x, y), (int(x), int(y)), (255, 0, 0), 1)
        return

    # 역행렬 계산 후 vt계산
    inverse_arr = np.linalg.inv(tmp)
    vt = np.dot(np.dot(inverse_arr, A.T), b)

    if np.linalg.norm(vt) == 0:
        # cv2.arrowedLine(img1, (x, y), (int(x), int(y)), (255, 0, 0), 1)
        return

    vt_n = vt / np.linalg.norm(vt) * 7

    np.nan_to_num(vt_n, copy=False)

    cv2.arrowedLine(img1, (x, y), (int(x + vt_n[1]), int(y + vt_n[0])), (0, 250, 0), 1)
    return


# pt1과 pt2를 화면에 표시
for x in range(0, width, 10):  # 1280
    for y in range(0, height, 10):  # 720
        lukasKanade(x, y)

cv2.imshow('result', img1)
cv2.waitKey()
cv2.destroyAllWindows()