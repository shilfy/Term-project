import cv2
import numpy as np

img = cv2.imread('289265818-30108bb8-9135-4704-a4f1-5b5e66b49fd3.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cat_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalcatface.xml')

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# 이미지에서 고양이 검출
cats = cat_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)


cat_count = 0
eye_count = 0

# 각 고양이에 대해 눈 검출
for (x, y, w, h) in cats:
    roi_gray = gray[y:y + h, x:x + w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    
    cat_count += 1
    eye_count += len(eyes)

    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(img[y:y + h, x:x + w], (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)


print(f"고양이 수: {cat_count}")
print(f"눈의 개수: {eye_count}")


cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
origin Code
