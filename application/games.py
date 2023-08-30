import cv2
import numpy as np
import time
import random
from typing import List, Tuple

from models import Res10, CascadeClassifier, YOLOv8, Own_model

SCREEN_HEIGHT = 720
SCREEN_WIDTH = 1280

RECT_HEIGHT = 250
RECT_WIDTH = 200

NUM_RECTANGLES = 10

BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

def detect_and_check(cropped_frame: np.ndarray, incorrect_count: int, correct_count: int, model) -> tuple[int, int, list[Tuple[int, int, int, int]], np.ndarray]:
    """
    Detects faces in a cropped frame and updates incorrect and correct counts.

    Args:
        cropped_frame (np.ndarray): Cropped frame for face detection.
        incorrect_count (int): Count of incorrect detections.
        correct_count (int): Count of correct detections.
        model: The face detection model.

    Returns:
        Tuple[int, int, List[Tuple[int, int, int, int]], np.ndarray]: Updated incorrect and correct counts, faces detected, and modified image.
    """
    faces, image = model.detect(cropped_frame)
    if len(faces) == 0:
        incorrect_count += 1
    else:
        correct_count += 1
    return incorrect_count, correct_count, faces, image

def play_1(model, difficulty: int = 1) -> tuple[list[tuple[int, int, int, int]], int, int]:
    """
    Plays the first game variant.

    Args:
        model: The face detection model.
        difficulty (int, optional): Difficulty level. Defaults to 1.

    Returns:
        Tuple[List[Tuple[int, int, int, int]], int, int]: List of answers and images, count of correct answers, count of incorrect answers.
    """
    cap = cv2.VideoCapture(0)
    cap.set(3, SCREEN_WIDTH)
    cap.set(4, SCREEN_HEIGHT)

    correct_count = 0
    incorrect_count = 0

    CHANGE_INTERVAL_DEPENDING_ON_DIFFICULTY = {
        '1': 10,
        '2': 5,
        '3': 3
    }

    next_change_time = time.time() + CHANGE_INTERVAL_DEPENDING_ON_DIFFICULTY[difficulty]
    rect_x = np.random.randint(0, SCREEN_WIDTH - RECT_WIDTH)
    rect_y = np.random.randint(0, SCREEN_HEIGHT - RECT_HEIGHT)
    rect_count = 0

    answers = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if rect_count < NUM_RECTANGLES:
            cv2.rectangle(frame, (rect_x, rect_y), (rect_x + RECT_WIDTH, rect_y + RECT_HEIGHT), BLUE, 4)

        current_time = time.time()

        if current_time >= next_change_time and rect_count < NUM_RECTANGLES:
            next_change_time = current_time + CHANGE_INTERVAL_DEPENDING_ON_DIFFICULTY[difficulty]

            copy_frame = frame.copy()
            cropped_frame = copy_frame[rect_y:rect_y + RECT_HEIGHT, rect_x:rect_x + RECT_WIDTH]
            incorrect_count, correct_count, faces, image_answer = detect_and_check(cropped_frame, incorrect_count, correct_count, model)
            answers.append((faces, image_answer))
            frame_height, frame_width, _ = frame.shape

            rect_x = np.random.randint(0, frame_width - RECT_WIDTH)
            rect_y = np.random.randint(0, frame_height - RECT_HEIGHT)
            rect_count += 1

        mirrored_frame = cv2.flip(frame, 1)

        if rect_count > 9:
            text = f"Click 'q' and see results."
            cv2.putText(mirrored_frame, text, org=(50, 150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2,
                        color=BLACK, thickness=2)

        cv2.imshow('Game Window', mirrored_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return answers, correct_count, incorrect_count

def play_2(model, difficulty: int = 1) -> float:
    """
    Plays the second game variant.

    Args:
        model: The face detection model.
        difficulty (int, optional): Difficulty level. Defaults to 1.

    Returns:
        float: Total game time.
    """
    def check_overlap(rect1: tuple[int, int, int, int], rect2: tuple[int, int, int, int]) -> bool:
        """
        Checks if two rectangles overlap.

        Args:
            rect1 (Tuple[int, int, int, int]): Coordinates of the first rectangle.
            rect2 (Tuple[int, int, int, int]): Coordinates of the second rectangle.

        Returns:
            bool: True if rectangles overlap, False otherwise.
        """
        start_x1, start_y1, end_x1, end_y1 = rect1
        start_x2, start_y2, end_x2, end_y2 = rect2
        if start_x2 >= start_x1 and start_y2 >= start_y1 and end_x2 <= end_x1 and end_y2 <= end_y1:
            return True
        return False

    cap = cv2.VideoCapture(0)
    cap.set(3, SCREEN_WIDTH)
    cap.set(4, SCREEN_HEIGHT)

    RECTANGLE_SCALE_DEPENDING_ON_DIFFICULTY = {
        '1': 1,
        '2': 0.8
    }

    ratio = RECTANGLE_SCALE_DEPENDING_ON_DIFFICULTY[difficulty]

    rectangles = []
    colors = []
    for _ in range(NUM_RECTANGLES):
        x = random.randint(0, SCREEN_WIDTH - RECT_WIDTH)
        y = random.randint(0, SCREEN_HEIGHT - RECT_HEIGHT)
        rectangles.append((x, y, x + int(ratio * RECT_WIDTH), y + int(ratio * RECT_HEIGHT)))
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        colors.append(color)

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image_to_detect = frame.copy()
        faces, _ = model.detect(image_to_detect)

        for face in faces:
            x1, y1, x2, y2 = face
            face_rect = (x1, y1, x2, y2)

            for idx, rect in enumerate(rectangles):
                if check_overlap(rect, face_rect):
                    del rectangles[idx]
                    del colors[idx]

        for rect, color in zip(rectangles, colors):
            x1, y1, x2, y2 = rect
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        mirrored_frame = cv2.flip(frame, 1)

        cv2.imshow('Face Game', mirrored_frame)

        if not rectangles:
            end_time = time.time()
            total_time = end_time - start_time
            cv2.destroyAllWindows()
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            total_time = None
            break

    cap.release()
    cv2.destroyAllWindows()

    return total_time
