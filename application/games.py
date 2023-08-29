import cv2
import numpy as np
import time
import random

from models import Res10, CascadeClassifier, YOLOv8, Own_model

SCREEN_HEIGHT = 720
SCREEN_WIDTH= 1280

RECT_HEIGHT = 250
RECT_WIDTH = 200

NUM_RECTANGLES = 10

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

def play_1(model, difficult=1):
    # Inicjalizacja kamery
    cap = cv2.VideoCapture(0)
    cap.set(3, SCREEN_WIDTH)
    cap.set(4, SCREEN_HEIGHT)

    # Liczniki poprawnych i błędnych detekcji
    correct_count = 0
    incorrect_count = 0

    if difficult == 1:
        change_interval = 10  # Interwał zmiany punktu (rect_x, rect_y) w sekundach
    elif difficult == 2:
        change_interval = 5  # Interwał zmiany punktu (rect_x, rect_y) w sekundach
    else:
        change_interval = 3  # Interwał zmiany punktu (rect_x, rect_y) w sekundach

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
        if rect_count < NUM_RECTANGLES:
            cv2.rectangle(frame, (rect_x, rect_y), (rect_x + RECT_WIDTH, rect_y + RECT_HEIGHT), BLUE, 4)

        current_time = time.time()

        if current_time >= next_change_time and rect_count < NUM_RECTANGLES:
            next_change_time = current_time + change_interval

            # tutaj robię detekcje twarzy
            copy_frame = frame.copy()
            cropped_frame = copy_frame[rect_y:rect_y + RECT_HEIGHT, rect_x:rect_x + RECT_WIDTH]
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


def play_2(model):
    # Funkcja sprawdzająca, czy dwie prostokąty na siebie nachodzą
    def check_overlap(rect1, rect2):
        start_x1, start_y1, end_x1, end_y1 = rect1
        start_x2, start_y2, end_x2, end_y2 = rect2
        if start_x2 >= start_x1 and start_y2 >= start_y1 and end_x2 <= end_x1 and end_y2 <= end_y1:
            return True
        return False

    # Inicjalizacja kamery
    cap = cv2.VideoCapture(0)
    cap.set(3, SCREEN_WIDTH)
    cap.set(4, SCREEN_HEIGHT)

    # Inicjalizacja zmiennych do gry
    rectangles = []
    colors = []
    for _ in range(NUM_RECTANGLES):
        x = random.randint(0, SCREEN_WIDTH - RECT_WIDTH)
        y = random.randint(0, SCREEN_HEIGHT - RECT_HEIGHT)
        rectangles.append((x, y, x + 200, y + 250))
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        colors.append(color)

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image_to_detect = frame.copy()
        # Detekcja twarzy
        faces, _ = model.detect(image_to_detect)

        for face in faces:
            x1, y1, x2, y2 = face
            face_rect = (x1, y1, x2, y2)

            for idx, rect in enumerate(rectangles):
                if check_overlap(rect, face_rect):
                    del rectangles[idx]
                    del colors[idx]

        # Rysowanie prostokątów
        for rect, color in zip(rectangles, colors):
            x1, y1, x2, y2 = rect
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        mirrored_frame = cv2.flip(frame, 1)

        cv2.imshow('Face Game', mirrored_frame)

        # Wyjście z pętli, gdy wszystkie prostokąty zniknęły
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