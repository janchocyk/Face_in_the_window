import cv2
import numpy as np
import time

import models
from models import *

SCREEN_HEIGHT = 720
SCREEN_WIDTH= 1280

def dettection_and_check(cropped_frame: np.ndarray, incorrect_count: int, correct_count: int, model) -> tuple[int, int]:
    faces, image = model.detection(cropped_frame)
    if len(faces) == 0:
        incorrect_count += 1
    else:
        correct_count += 1
    return incorrect_count, correct_count, faces, image

def prepare_answer(faces: list[tuple[int, int, int, int]], image_answer: np.ndarray):
    if len(faces) == 0:
        x, y, w, h = (0, 0, 200, 250)
        answer = cv2.rectangle(image_answer, (x, y), (x + w, y + h), (0, 0, 255), 8)
        text = "Incorrect"
        ready_answer = cv2.putText(image_answer, text, org=(30, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 255), thickness=2)

    else:
        text = "Correct"
        ready_answer = cv2.putText(image_answer, text, org=(30, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 0), thickness=2)
    return ready_answer

def play(model):
    # Inicjalizacja kamery
    cap = cv2.VideoCapture(0)
    cap.set(3, SCREEN_WIDTH)
    cap.set(4, SCREEN_HEIGHT)

    # Liczniki poprawnych i błędnych detekcji
    correct_count = 0
    incorrect_count = 0

    frame_duration = 3  # Czas trwania ramki w sekundach
    change_interval = 3  # Interwał zmiany punktu (rect_x, rect_y) w sekundach
    max_rectangles = 10  # Maksymalna liczba wyświetleń prostokątów
    next_change_time = time.time() + change_interval
    rect_width = 200
    rect_height = 250
    rect_x = np.random.randint(0, 1280 - rect_width)
    rect_y = np.random.randint(0, 720 - rect_height)
    rect_count = 0

    answers = []
    answers_to_display = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Rysowanie prostokąta na obrazie
        if rect_count < max_rectangles:
            cv2.rectangle(frame, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (255, 0, 0), 4)

        current_time = time.time()

        if current_time >= next_change_time and rect_count < max_rectangles:
            next_change_time = current_time + change_interval

            # tutaj robię detekcje twarzy
            cropped_frame = frame[rect_y:rect_y + rect_height, rect_x:rect_x + rect_width]
            incorrect_count, correct_count, faces, image_answer = dettection_and_check(cropped_frame, incorrect_count, correct_count, model)
            # if len(faces) == 0:
            #     cv2.rectangle(frame, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (0, 0, 255), -1)
            # else:
            #     cv2.rectangle(frame, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (0, 255, 0), -1)
            answers.append((faces, image_answer))
            frame_height, frame_width, _ = frame.shape

            # Losowanie nowego punktu (rect_x, rect_y)
            rect_x = np.random.randint(0, frame_width - 200)
            rect_y = np.random.randint(0, frame_height - 250)
            rect_count += 1



        mirrored_frame = cv2.flip(frame, 1)
        if rect_count > 9:
            text = f"Click 'q' and see results."
            cv2.putText(mirrored_frame, text, org=(50, 150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2,
                        color=(255, 0, 0), thickness=2)
        cv2.imshow('Game Window', mirrored_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Wciśnięcie 'q' zamyka okno
            break

    cap.release()
    cv2.destroyAllWindows()
    return answers, correct_count, incorrect_count

def show_answers(answers, correct_count, incorrect_count):
    answers_to_display = []
    for idx, a in enumerate(answers):
        face, image_answer = answers[idx]
        ready_answer = prepare_answer(face, image_answer)
        answers_to_display.append(ready_answer)

    # Tworzenie pustego obrazu o kolorze błękitnym
    main_image = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
    main_image[:, :] = (173, 216, 230)  # Kolor błękitny w formacie BGR

    image_width = 200
    image_height = 250

    # Liczba obrazków w rzędzie i kolumnie
    num_rows = 2
    num_columns = 5

    # Puste miejsce między obrazkami
    spacing = 10

    # Sprawdzenie, czy liczba obrazków do wyświetlenia jest zgodna z liczbą komórek
    if len(answers_to_display) == num_rows * num_columns:
        for row in range(num_rows):
            for col in range(num_columns):
                # Współrzędne lewego górnego rogu obrazka na obrazie głównym
                x = 115 + col * (image_width + spacing)
                y = 180 + row * (image_height + spacing)

                # Pobranie obrazka z tablicy 'answers_to_display'
                image = answers_to_display[row * num_columns + col]

                # Zmiana rozmiaru obrazka na 250x200 (jeśli jest inny)
                if image.shape[:2] != (image_height, image_width):
                    image = cv2.resize(image, (image_width, image_height))

                # Umieszczenie obrazka na obrazie głównym
                main_image[y:y + image_height, x:x + image_width] = image

        text = f"SUMMARY: Correct {correct_count}, Incorrect {incorrect_count}"
        cv2.putText(main_image, text, org=(380, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                    color=(255, 0, 0), thickness=2)
        # Wyświetlanie obrazu głównego
        cv2.imshow('Results', main_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        raise ValueError


def main():
    # settings
    model = models.Res10()
    # playing
    answers, correct_count, incorrect_count = play(model)
    show_answers(answers, correct_count, incorrect_count)

if __name__ == '__main__':
    main()
