import cv2
import numpy as np
import time

from models import *

SCREEN_HEIGHT = 720
SCREEN_WIDTH= 1280

RECT_HEIGHT = 250
RECT_WIDTH = 200

BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

def detect_and_check(cropped_frame: np.ndarray, incorrect_count: int, correct_count: int, model) -> tuple[int, int]:
    faces, image = model.detect(cropped_frame)
    if len(faces) == 0:
        incorrect_count += 1
    else:
        correct_count += 1
    return incorrect_count, correct_count, faces, image

def play_1(model):
    # Inicjalizacja kamery
    cap = cv2.VideoCapture(0)
    cap.set(3, SCREEN_WIDTH)
    cap.set(4, SCREEN_HEIGHT)

    # Liczniki poprawnych i błędnych detekcji
    correct_count = 0
    incorrect_count = 0

    change_interval = 5  # Interwał zmiany punktu (rect_x, rect_y) w sekundach
    max_rectangles = 10  # Maksymalna liczba wyświetleń prostokątów
    next_change_time = time.time() + change_interval
    rect_x = np.random.randint(0, SCREEN_WIDTH - RECT_WIDTH)
    rect_y = np.random.randint(0, SCREEN_HEIGHT - RECT_HEIGHT)
    rect_count = 0

    answers = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Rysowanie prostokąta na obrazie
        if rect_count < max_rectangles:
            cv2.rectangle(frame, (rect_x, rect_y), (rect_x + RECT_WIDTH, rect_y + RECT_HEIGHT), BLUE, 4)

        current_time = time.time()

        if current_time >= next_change_time and rect_count < max_rectangles:
            next_change_time = current_time + change_interval

            # tutaj robię detekcje twarzy
            cropped_frame = frame[rect_y:rect_y + RECT_HEIGHT, rect_x:rect_x + RECT_WIDTH]
            incorrect_count, correct_count, faces, image_answer = detect_and_check(cropped_frame, incorrect_count, correct_count, model)
            answers.append((faces, image_answer))
            frame_height, frame_width, _ = frame.shape

            # Losowanie nowego punktu (rect_x, rect_y)
            rect_x = np.random.randint(0, frame_width - RECT_WIDTH)
            rect_y = np.random.randint(0, frame_height - RECT_HEIGHT)
            rect_count += 1



        mirrored_frame = cv2.flip(frame, 1)
        if rect_count > 9:
            text = f"Click 'q' and see results."
            cv2.putText(mirrored_frame, text, org=(50, 150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2,
                        color=BLACK, thickness=2)
        cv2.imshow('Game Window', mirrored_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Wciśnięcie 'q' zamyka okno
            break

    cap.release()
    cv2.destroyAllWindows()
    return answers, correct_count, incorrect_count


def play_2():
    print('Play 2 comming soon!')