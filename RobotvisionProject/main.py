import cv2
import numpy as np

img1_origin = cv2.imread('frame1.jpg')
img2_origin = cv2.imread('frame2.jpg')

img1 = cv2.resize(img1_origin, dsize=(1280, 720), interpolation=cv2.INTER_AREA)
img2 = cv2.resize(img2_origin, dsize=(1280, 720), interpolation=cv2.INTER_AREA)

height, width, _ = img1.shape

# 그레이스케일로 변환
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

# 코너점 찾는 함수, 그레이스케일 영상만 입력 가능
#pt1 = cv2.goodFeaturesToTrack(gray1, 50, 0.01, 10)

# 찾은 코너점 정보를 옵티컬플로우 함수에 입력
# src1, src2에서 움직임 정보를 찾아내고 pt1에 입력한 좌표가 어디로 이동했는지 파악
#pt2, status, err = cv2.calcOpticalFlowPyrLK(img1, img2, pt1, None)

# 가중합으로 개체가 어느 정도 이동했는지 보기 위함
#dst = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)

def lukasKanade(img):




# pt1과 pt2를 화면에 표시
for x in range(width):
    if x % 10 != 0:
        continue
    for y in range(height):
        # pt1과 pt2를 이어주는 선 그리기
        if y % 10 == 0:
            cv2.circle(img1, (x, y), 1, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.arrowedLine(img1, (x,y), (x+5, y+5), (255, 255, 255), 1)
        else:
            continue

cv2.imshow('result', img1)
cv2.waitKey()
cv2.destroyAllWindows()

