from utils.cvfpscalc import CvFpsCalc
import numpy as np

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []

    for lm in landmarks.landmark:
        x = min(int(lm.x * image_width), image_width - 1)
        y = min(int(lm.y * image_height), image_height - 1)
        landmark_point.append([x, y])

    return landmark_point

def pre_process_landmark(landmark_list):
    base_x, base_y = landmark_list[0]

    relative_landmarks = []
    for x, y in landmark_list:
        relative_landmarks.append(x - base_x)
        relative_landmarks.append(y - base_y)

    max_value = max(list(map(abs, relative_landmarks)))
    normalized = list(map(lambda x: x / max_value, relative_landmarks))
    return normalized
