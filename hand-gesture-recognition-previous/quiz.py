#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import random
import csv
import copy
import itertools
import os
import glob

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from model import KeyPointClassifier


# ===== Helper functions =====

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for lm in landmarks.landmark:
        landmark_x = min(int(lm.x * image_width), image_width - 1)
        landmark_y = min(int(lm.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point


def pre_process_landmark(landmark_list):
    temp = copy.deepcopy(landmark_list)
    base_x, base_y = temp[0]
    for p in temp:
        p[0] -= base_x
        p[1] -= base_y
    temp = list(itertools.chain.from_iterable(temp))
    max_value = max(list(map(abs, temp))) or 1
    return [v / max_value for v in temp]


def draw_landmarks(image, landmark_point):
    for x, y in landmark_point:
        cv.circle(image, (x, y), 4, (0, 255, 0), -1)
    return image


# ===== Quiz Program =====

def main():

    # ---------------- Ask user how many rounds ----------------
    while True:
        try:
            num_questions = int(input("How many quiz rounds? "))
            if num_questions > 0:
                break
        except:
            pass
        print("Please enter a positive integer.")

    # ---------------- Load ASL label list ----------------
    with open("model/keypoint_classifier/keypoint_classifier_label.csv", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        labels = [row[0] for row in reader]

    # Filter to letters only
    available_letters = [lbl for lbl in labels if len(lbl) == 1 and lbl.isalpha()]
    print("Quiz will use these letters:", ", ".join(available_letters))

    # ---------------- Load ASL reference images ----------------
    letters_dir = os.path.join(os.path.dirname(__file__), "ASL_Letters")
    asl_images = {}

    for letter in available_letters:
        pattern = os.path.join(letters_dir, f"ASL_{letter}.*")
        candidates = glob.glob(pattern)

        if not candidates:
            print(f"[WARN] No image found for {letter}")
            continue

        img_path = candidates[0]
        img = cv.imread(img_path)

        if img is None:
            print(f"[WARN] Unable to read: {img_path}")
            continue

        img = cv.resize(img, (600, 600), interpolation=cv.INTER_LINEAR)
        asl_images[letter] = img
        print(f"[INFO] Loaded ASL image for {letter}")

    # ---------------- Camera + Mediapipe ----------------
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 540)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    keypoint_classifier = KeyPointClassifier()
    fps_calc = CvFpsCalc(buffer_len=10)

    correct_count = 0
    previous_letter = None

    # ---------------- Quiz Loop ----------------
    for q in range(1, num_questions + 1):

        # pick letter (no immediate repeat)
        while True:
            target_letter = random.choice(available_letters)
            if target_letter != previous_letter:
                break
        previous_letter = target_letter

        print(f"\nQuestion {q}/{num_questions}")

        # ---------------- READY SCREEN ----------------
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera failure.")
                cap.release()
                cv.destroyAllWindows()
                return

            frame = cv.flip(frame, 1)
            ready_img = frame.copy()

            cv.putText(
                ready_img,
                "When ready, press any key to start",
                (30, 80),
                cv.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2,
            )

            cv.putText(
                ready_img,
                f"Question {q}/{num_questions}",
                (30, 140),
                cv.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
            )

            cv.imshow("ASL Quiz", ready_img)

            key = cv.waitKey(1) & 0xFF
            if key == 27:  # ESC aborts quiz
                print("Quiz aborted.")
                cap.release()
                cv.destroyAllWindows()
                return
            if key != 255:  # any non-ESC key
                break

        # ---------------- START QUESTION ----------------
        print(f"Target letter is '{target_letter}'")
        start_time = time.time()
        predicted_label = None

        while True:
            fps = fps_calc.get()
            ret, frame = cap.read()
            if not ret:
                print("Camera failure.")
                cap.release()
                cv.destroyAllWindows()
                return

            frame = cv.flip(frame, 1)
            debug = frame.copy()

            # Mediapipe
            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand in results.multi_hand_landmarks:
                    lm_list = calc_landmark_list(debug, hand)
                    pre = pre_process_landmark(lm_list)
                    idx = keypoint_classifier(pre)
                    predicted_label = labels[idx]
                    draw_landmarks(debug, lm_list)

            elapsed = time.time() - start_time

            # ---------- CORRECT ----------
            if predicted_label == target_letter:
                correct_count += 1

                cv.putText(
                    debug,
                    f"Correct: {target_letter}",
                    (50, 90),
                    cv.FONT_HERSHEY_SIMPLEX,
                    2.0,
                    (0, 255, 0),
                    4,
                )

                cv.putText(
                    debug,
                    "Press any key for next question",
                    (50, 160),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),
                    2,
                )

                cv.imshow("ASL Quiz", debug)
                cv.waitKey(0)
                break

            # ---------- INCORRECT / TIMEOUT ----------
            if elapsed > 10:
                print(f"Incorrect — correct letter was {target_letter}")

                feedback = debug.copy()

                cv.putText(
                    feedback,
                    f"Incorrect: {target_letter}",
                    (50, 90),
                    cv.FONT_HERSHEY_SIMPLEX,
                    2.0,
                    (0, 0, 255),
                    4,
                )

                cv.putText(
                    feedback,
                    "Press any key for next question",
                    (50, 160),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),
                    2,
                )

                cv.imshow("ASL Quiz", feedback)

                # Show reference ASL image
                if target_letter in asl_images:
                    cv.imshow("Correct ASL Sign", asl_images[target_letter])

                cv.waitKey(0)
                cv.destroyWindow("Correct ASL Sign")
                break

            # ---------- HUD ----------
            cv.putText(debug, f"Target: {target_letter}", (10, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv.putText(debug, f"Time left: {max(0,10-int(elapsed))}s", (10, 70),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv.imshow("ASL Quiz", debug)

            if cv.waitKey(1) & 0xFF == 27:
                print("Quiz aborted.")
                cap.release()
                cv.destroyAllWindows()
                return

    # ---------------- FINAL SCORE ----------------
    cap.release()
    cv.destroyAllWindows()

    score = (correct_count / num_questions) * 100.0
    print("\nQuiz completed!")
    print(f"Score: {correct_count}/{num_questions} = {score:.1f}%")

    score_img = np.zeros((400, 600, 3), dtype=np.uint8)
    cv.putText(score_img, "Final Score:", (50, 120),
               cv.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    cv.putText(score_img,
               f"{correct_count}/{num_questions}  ({score:.1f}%)",
               (50, 220),
               cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv.putText(score_img, "Press any key to close",
               (130, 300), cv.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)

    cv.imshow("Your Score", score_img)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
