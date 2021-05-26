import cv2
import numpy as np

img_origin = cv2.imread('frame1.jpg')
img = cv2.resize(img_origin, dsize=(510, 250), interpolation=cv2.INTER_AREA)
height, width, _ = img.shape  # 360(y) 640(x)
B, G, R = cv2.split(img)

X = np.zeros((250, 510, 5))  # 모든 좌표의 y, x, r, g, b를 저장해놓은 배열
V = np.zeros((width * height, 5))   # 모든 픽셀에 대해 v를 구하고 이를 이용해 클러스터링 진행
yt_variation_min = 3


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


# yt_next을 구하기 위한 함수. 원형 커널 내의 mean을 구함.
def mean_shift(yt):
    hr = 10
    hs = 4
    num = 0
    numerator = 0
    denominator = 0
    yt_next = yt

    yt_loc = np.array([yt[0], yt[1]])
    yt_rgb = np.array([yt[2], yt[3], yt[4]])
    list_satisfied = np.zeros((width * height, 5))

    while True:

        # 모든 픽셀을 돌면서 hr, hs를 만족하는 것들을 찾아라.
        for x in range(0, width):
            for y in range(0, height):

                # spatial(xi_loc)과 rgb(xi_rgb) 관련된 두 부분으로 나누어 확인
                xi_loc = np.array([X[y][x][0], X[y][x][1]])
                xi_rgb = np.array([X[y][x][2], X[y][x][3], X[y][x][4]])
                
                # 두 부분 다 지정한 h 커널 내에 존재할 경우 list에 추가
                if distance(yt_loc, xi_loc) < hs and distance(yt_rgb, xi_rgb) < hr:
                    list_satisfied[num] = X[y][x]
                    num = num + 1
                else:
                    continue

        # list에 있는 xi를 전부 확인하며 mean을 구함
        for n in range(0, num):
            list_satisfied_loc = np.array([list_satisfied[num][0], list_satisfied[num][1]])
            list_satisfied_rgb = np.array([list_satisfied[num][2], list_satisfied[num][3], list_satisfied[num][4]])
            k_loc = gaussian_kernel(list_satisfied_loc, yt_loc, hs)
            k_rgb = gaussian_kernel(list_satisfied_rgb, yt_rgb, hr)
            k = k_loc * k_rgb


            # yt_next 구하기 위한 시그마 계산
            numerator = numerator + list_satisfied[num] * k     # 분자 (numerator)
            denominator = denominator + k                       # 분모 (denominator)

        yt = yt_next
        yt_next = numerator / denominator
        print(numerator, denominator)

        if distance(yt, yt_next) < yt_variation_min:
            print(yt_next)
            break


for x in range(0, width):
    for y in range(0, height):
        X[y][x][0] = x
        X[y][x][1] = y
        X[y][x][2] = R[y][x]
        X[y][x][3] = G[y][x]
        X[y][x][4] = B[y][x]

# 모든 픽셀에 대해 mean shift 진행
for x in range(0, width):
    for y in range(0, height):
        yt = X[y][x]
        mean_shift(yt)

cv2.imshow('result', img)
cv2.waitKey()
cv2.destroyAllWindows()