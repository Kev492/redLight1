import cv2
import numpy as np
import os

# 결과 이미지를 저장할 폴더 경로
output_folder = 'C:/pyworkspace1/captured/trafficLight/trafficLightImage/greenLight'

# 이미지 불러오기
image_path = 'C:/pyworkspace1/captured/trafficLight/trafficLightImage/image3.jpeg'

target_image = cv2.imread('C:/pyworkspace1/captured/trafficLight/trafficLightImage/greenPerson1.jpeg')

# 파일 이름을 동적으로 생성
existing_files = os.listdir(output_folder)
num_existing_files = len(existing_files) + 1
result_image_filename = f'extractedImage{num_existing_files}.jpeg'
result_image_path = os.path.join(output_folder, result_image_filename)

# 이미지 불러오기
image = cv2.imread(image_path)

# 녹색 범위
lower_green = np.array([40, 40, 40])
upper_green = np.array([90, 255, 255])

# 이미지를 HSV로 변환
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 초록색 객체 검출
green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

# 객체 검출된 마스크를 이용하여 원본 이미지에 객체 표시
result = cv2.bitwise_and(image, image, mask=green_mask)

# Find contours of green objects
contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort the contours by area in descending order to get the largest ones first
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]  # Get the largest 10 contours

# Initialize variables for the best match
best_match = None
best_match_score = float('-inf')

# Scale factors to resize the target image
scale_factors = [0.5, 0.75, 1.0, 1.25, 1.5]

for scale_factor in scale_factors:
    try:
        # Resize the target image
        scaled_target = cv2.resize(target_image, None, fx=scale_factor, fy=scale_factor)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            green_object = image[y:y + h, x:x + w]  # Extract the green object

            # Resize the green object to match the size of the scaled target image
            scaled_red_object = cv2.resize(green_object, (scaled_target.shape[1], scaled_target.shape[0]))

            # Compare the scaled green object with the scaled target image using cv2.matchTemplate with TM_CCOEFF_NORMED method
            result = cv2.matchTemplate(scaled_red_object, scaled_target, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            # If the current rectangle is a better match, update best_match and best_match_score
            if max_val > best_match_score:
                best_match = (x, y, w, h)
                best_match_score = max_val
                best_scale_factor = scale_factor

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if (x, y, w, h) == best_match:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 빨간색 사각형
            else:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)  # 노란색 사각형

        if best_match is not None:
            print(f"Best match found at scale factor {best_scale_factor}")
            # Save the result image
            cv2.imwrite(result_image_path, image)
        else:
            print("None best match!")

    except cv2.error:
        print(f"스케일 팩터 {scale_factor}에서 오류 발생. 계속 진행합니다.")