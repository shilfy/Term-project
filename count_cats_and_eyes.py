import cv2
import numpy as np


# 이미지 로드
image = cv2.imread('image path')
# 이미지를 그레이스케일로 변환
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 얼굴 및 눈 검출을 위한 미리 훈련된 Haarcascades 로드
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalcatface.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# 이미지에서 얼굴 검출
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3)


cat_count = 0
eye_count = 0

# 검출된 얼굴들에 대해 반복
for (x, y, w, h) in faces:

# 눈 검출을 위한 관심 영역(ROI) 추출
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = image[y:y+h, x:x+w]

    # 얼굴 ROI에서 눈 검출
    eyes = eye_cascade.detectMultiScale(roi_gray)

    # 고양이의 개수 세기
    cat_count += 1
    # 눈의 개수 세기
    eye_count += len(eyes)

# 얼굴 주변에 사각형 그리기
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

# 눈 주변에 사각형 그리기
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(image, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)

# 개수 출력
print(f"고양이 수: {cat_count}")
print(f"눈의 개수: {eye_count}")

# 결과 표시
cv2.imshow('img', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
