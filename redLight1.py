import cv2
import numpy as np
import os

# 결과 이미지를 저장할 폴더 경로
output_folder = 'C:/pyworkspace1/captured/trafficLight/trafficLightImage/redLight'

# 이미지 불러오기
image_path = 'C:/pyworkspace1/captured/trafficLight/trafficLightImage/image1.jpeg'

target_image = cv2.imread('C:/pyworkspace1/captured/trafficLight/trafficLightImage/redPerson1.jpeg')

# 파일 이름을 동적으로 생성
existing_files = os.listdir(output_folder)
num_existing_files = len(existing_files) + 1
result_image_filename = f'extractedImage{num_existing_files}.jpeg'
result_image_path = os.path.join(output_folder, result_image_filename)

# 이미지 불러오기
image = cv2.imread(image_path)

# 빨간색 범위를 정의
lower_red1 = np.array([0, 100, 100])    # HSV 공간에서의 빨강 범위 (하한)
upper_red1 = np.array([15, 255, 255])

lower_red2 = np.array([160, 100, 100])  # HSV 공간에서의 빨강 범위 (상한)
upper_red2 = np.array([180, 255, 255])

# 이미지를 HSV로 변환
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 빨간색 객체 검출
red_mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
red_mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
red_mask = red_mask1 + red_mask2  # 두 범위의 합

# 객체 검출된 마스크를 이용하여 원본 이미지에 객체 표시
result = cv2.bitwise_and(image, image, mask=red_mask)

# Find contours of red objects
contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort the contours by area in descending order to get the largest ones first
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]  # Get the largest 10 contours

# Initialize variables for the best match
best_match = None
best_match_score = float('-inf')

# Resize the target image to match the size of the largest red object
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    red_object = image[y:y + h, x:x + w]  # Extract the red object
    scaled_target = cv2.resize(target_image, (red_object.shape[1], red_object.shape[0]))

    # Compare the red object with the scaled target image using cv2.matchTemplate with TM_CCOEFF_NORMED method
    result = cv2.matchTemplate(red_object, scaled_target, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    # If the current rectangle is a better match, update best_match and best_match_score
    if max_val > best_match_score:
        best_match = (x, y, w, h)
        best_match_score = max_val

# Draw the rectangles on the red_object
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if (x, y, w, h) == best_match:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 초록색 사각형
    else:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)  # 노란색 사각형

if best_match is not None:
    print(f"Best match found")
    print("Score: " + str(best_match_score))
    # Save the result image
    cv2.imwrite(result_image_path, image)
else:
    print("None best match!")