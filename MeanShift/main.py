import cv2
import numpy as np

img_origin = cv2.imread('frame1.jpg')
img = cv2.resize(img_origin, dsize=(510, 250), interpolation=cv2.INTER_AREA)

height, width, _ = img.shape  # 360(y) 640(x)
B, G, R = cv2.split(img)

X = np.zeros((510, 250, 5))  # y, x, rgb

yt_variation_min = 0.1

# 이미지 크기 밖으로 index가 벗어나는지 확인
def isOutOfIndex(x, y):
    if x < 0 or x > width-1 or y < 0 or y > height-1:
        return False
    else:
        return True

# 벡터 사이 거리를 구하기 위해
def distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


# yt와 xi의 거리를 구하기 위해 가우시안 커널을 사용
def gaussian_kernel(xi, yt, h):
    return np.exp(-((distance(xi, yt) / h) ** 2))


# y_t+1을 구하기 위한 함수. 원형 커널 내의 mean을 구함.
def circle_kernel_mean(yt):
    hr = 10
    hs = 4

    yt_loc = np.array([yt[0], yt[1]])
    yt_rgb = np.array([yt[3], yt[4], yt[5]])
    list_satisfied = np.zeros((510, 250, 5))

    for x in range(0, width):
        for y in range(0, height):
            # 모든 픽셀을 돌면서 hr, hs를 만족하는 것들을 찾아라.

    num = 0
    for x in range(yt_loc[0] - hs, yt_loc[0] + hs):
        for y in range(yt_loc[1] - hs, yt_loc[1] + hs):
            if isOutOfIndex(x, y):
                tmp_array = np.array(x, y)
                if distance(tmp_array, yt_loc) < hs:
                    list_loc[num][0] = x
                    list_loc[num][1] = y
                    num = num + 1

    num = 0
    for x in range(yt_rgb[0] - hr, yt_rgb[0] + hr):
        for y in range(yt_rgb[1] - hr, yt_rgb[1] + hr):
            for z in range(yt_rgb[2] - hr, yt_rgb[2] + hr):
                if isOutOfIndex(x, y):
                    tmp_array = np.array(x, y, z)
                    if distance(tmp_array, yt_rgb) < hr:
                        list_rgb[num][0] = x
                        list_rgb[num][1] = y
                        list_rgb[num][2] = z
                        num = num + 1


    xi_loc = np.array(xi[0], xi[1])
    xi_rbh = np.array(xi[3], xi[4], yt[5])

    k_loc = gaussian_kernel(xi_loc, yt_loc, hs)
    k_rgb = gaussian_kernel(xi_rgb, yt_rgb, hr)


# 데이터 분포를 바탕으로 Mean Shift 진행
def meanShift(x, y):
    yt = np.array([x, y, R[y, x], G[y, x], B[y, x]])

    while True:
        yt_next = circle_kernel_mean(yt)
        if distance(yt_next, yt) < yt_variation_min:
            break

    return

for x in range(0, width):
    for y in range(0, height):
        X[y][x][0] = x
        X[y][x][1] = y
        X[y][x][2] = R[y][x]
        X[y][x][3] = G[y][x]
        X[y][x][4] = B[y][x]

for x in (0, width - 1):
    for y in (0, height - 1):
        meanShift(x, y)

cv2.imshow('result', img)
cv2.waitKey()
cv2.destroyAllWindows()