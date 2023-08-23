import cv2
import numpy as np
import tkinter as tk

from models import Res10, CascadeClassifier
from application.games import play_1, play_2, SCREEN_HEIGHT, SCREEN_WIDTH, RECT_HEIGHT, RECT_WIDTH, RED, BLACK, WHITE

game_variant = None
chosen_model = 'Res10'

# Funkcja obsługująca naciśnięcie przycisku "Gra 1"
def game1_button_callback():
    global game_variant
    game_variant = 1
    game1_button.config(state=tk.DISABLED)  # Zmiana koloru tła przycisku i dezaktywacja
    game2_button.config(state=tk.NORMAL)  # Zmiana koloru tła przycisku i dezaktywacja

# Funkcja obsługująca naciśnięcie przycisku "Gra 2"
def game2_button_callback():
    global game_variant
    game_variant = 2
    game2_button.config(state=tk.DISABLED)  # Zmiana koloru tła przycisku i dezaktywacja
    game1_button.config(state=tk.NORMAL)  # Zmiana koloru tła przycisku i dezaktywacja

def model1_button_callback():
    global chosen_model
    chosen_model = 'Res10'
    model1_button.config(state=tk.DISABLED)  # Zmiana koloru tła przycisku i dezaktywacja
    model2_button.config(state=tk.NORMAL)  # Zmiana koloru tła przycisku i dezaktywacja

def model2_button_callback():
    global chosen_model
    chosen_model = 'Casscade_Clasiffer'
    model2_button.config(state=tk.DISABLED)  # Zmiana koloru tła przycisku i dezaktywacja
    model1_button.config(state=tk.NORMAL)  # Zmiana koloru tła przycisku i dezaktywacja
def confirm_button_callback():
    root.quit()

def set_the_game():
    global game1_button, game2_button, model1_button, model2_button, root

    # Tworzenie okna tkinter
    root = tk.Tk()
    root.title("Ustawienia Gry")

    # Tekst powitalny i instrukcje
    welcome_label = tk.Label(root, text="Witaj w grze!")
    instructions_label = tk.Label(root, text="Wybierz wariant gry i wpisz swoje imię (opcjonalnie).")
    welcome_label.pack()
    instructions_label.pack()

    # Pole do wpisania imienia (opcjonalnie)
    name_label = tk.Label(root, text="Wpisz swoje imię:")
    name_entry = tk.Entry(root)
    name_label.pack()
    name_entry.pack()

    # Przyciski wyboru wariantu gry
    game_variants = tk.Label(root, text="Wybierz rozgrywkę:")
    game1_button = tk.Button(root, text="Gra 1", command=game1_button_callback)
    game2_button = tk.Button(root, text="Gra 2", command=game2_button_callback)
    game_variants.pack()
    game1_button.pack()
    game2_button.pack()

    # Przyciski wyboru modelu
    model_variants = tk.Label(root, text="Wybierz model detekcji, który ma być użyty:")
    model1_button = tk.Button(root, text="Model Res10", command=model1_button_callback)
    model2_button = tk.Button(root, text="Model Kaskadowy", command=model2_button_callback)
    model_variants.pack()
    model1_button.pack()
    model2_button.pack()

    # Przycisk zatwierdzenia
    confirm_set = tk.Label(root, text="Jeśli jesteś gotowy to zatwierdź ustawienia:")
    confirm_button = tk.Button(root, text="Zatwierdź", command=confirm_button_callback)
    confirm_set.pack()
    confirm_button.pack()

    # Główna pętla interfejsu tkinter
    root.mainloop()

    if name_entry.get():
        player_name = name_entry.get()
    else:
        player_name = 'Player'

    return player_name, game_variant, chosen_model

def prepare_answer(faces: list[tuple[int, int, int, int]], image_answer: np.ndarray):
    if len(faces) == 0:
        x, y, w, h = (0, 0, RECT_WIDTH, RECT_HEIGHT)
        answer = cv2.rectangle(image_answer, (x, y), (x + w, y + h), RED, 8)
        text = "Incorrect"
        ready_answer = cv2.putText(image_answer, text, org=(30, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=RED, thickness=2)

    else:
        text = "Correct"
        ready_answer = cv2.putText(image_answer, text, org=(30, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=BLACK, thickness=2)
    return ready_answer


def show_answers(answers, correct_count, incorrect_count, player_name):
    answers_to_display = []
    for idx, a in enumerate(answers):
        face, image_answer = answers[idx]
        ready_answer = prepare_answer(face, image_answer)
        answers_to_display.append(ready_answer)

    # Tworzenie pustego obrazu o kolorze błękitnym
    main_image = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
    main_image[:, :] = WHITE

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
                x = 115 + col * (RECT_WIDTH + spacing)
                y = 180 + row * (RECT_HEIGHT + spacing)

                # Pobranie obrazka z tablicy 'answers_to_display'
                image = answers_to_display[row * num_columns + col]

                # Zmiana rozmiaru obrazka na 250x200 (jeśli jest inny)
                if image.shape[:2] != (RECT_HEIGHT, RECT_WIDTH):
                    image = cv2.resize(image, (RECT_WIDTH, RECT_HEIGHT))

                # Umieszczenie obrazka na obrazie głównym
                main_image[y:y + RECT_HEIGHT, x:x + RECT_WIDTH] = image

        text = f"SUMMARY for {player_name}: Correct {correct_count}, Incorrect {incorrect_count}"
        cv2.putText(main_image, text, org=(380, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                    color=BLACK, thickness=2)
        # Wyświetlanie obrazu głównego
        cv2.imshow('Results', main_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        raise ValueError


def main():
    # settings
    player_name, game_variant, chosen_model = set_the_game()
    if chosen_model == 'Res10':
        model = Res10()
    else:
        model = CascadeClassifier()
    # playing
    if game_variant == 1:
        answers, correct_count, incorrect_count = play_1(model)
        show_answers(answers, correct_count, incorrect_count, player_name)
    elif game_variant == 2:
        play_2()

if __name__ == '__main__':
    main()
