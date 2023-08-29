import cv2
import numpy as np
import tkinter as tk

from models import Res10, CascadeClassifier, YOLOv8, Own_model
from games import play_1, play_2, SCREEN_HEIGHT, SCREEN_WIDTH, RECT_HEIGHT, RECT_WIDTH, RED, BLACK, WHITE


game_variant = 1
difficult_level = 1


def set_the_game():
    global game_variant, difficult_level


    def clear_frame(frame):
        for widget in frame.winfo_children():
            widget.destroy()


    def create_game1_frame():
        game2_frame.pack_forget()
        game1_frame.pack()
        difficult1_level_label = tk.Label(game1_frame, text="Ustaw poziom trudności:")
        # Tworzenie trzech radio buttonów
        radio1_button1 = tk.Radiobutton(game1_frame, text="Easy", variable=selected_option1, value=1, command=update_difficulty1)
        radio1_button2 = tk.Radiobutton(game1_frame, text="Medium", variable=selected_option1, value=2, command=update_difficulty1)
        radio1_button3 = tk.Radiobutton(game1_frame, text="Hard", variable=selected_option1, value=3, command=update_difficulty1)
        selected_option1.set(None)
        difficult1_level_label.pack()
        radio1_button1.pack()
        radio1_button2.pack()
        radio1_button3.pack()


    def create_game2_frame():
        game1_frame.pack_forget()
        game2_frame.pack()
        difficult2_level_label = tk.Label(game2_frame, text="Ustaw wielkość ramek:")
        # Tworzenie trzech radio buttonów
        radio2_button1 = tk.Radiobutton(game2_frame, text="Small", variable=selected_option2, value=1, command=update_difficulty2)
        radio2_button2 = tk.Radiobutton(game2_frame, text="Normal", variable=selected_option2, value=2, command=update_difficulty2)
        selected_option2.set(None)
        difficult2_level_label.pack()
        radio2_button1.pack()
        radio2_button2.pack()


    def game1_button_callback():
        global game_variant
        game_variant = 1
        game1_button.config(state=tk.DISABLED)
        game2_button.config(state=tk.NORMAL)
        clear_frame(game2_frame)
        create_game1_frame()


    def game2_button_callback():
        global game_variant
        game_variant = 2
        game2_button.config(state=tk.DISABLED)
        game1_button.config(state=tk.NORMAL)
        clear_frame(game1_frame)
        create_game2_frame()


    def update_difficulty1():
        global difficult_level
        difficult_level = selected_option1.get()


    def update_difficulty2():
        global difficult_level
        difficult_level = selected_option2.get()


    def confirm_button_callback():
        root.quit()


    root = tk.Tk()  # Tworzenie głównego okna
    selected_option1 = tk.StringVar(value=None)
    selected_option2 = tk.StringVar(value=None)

    root.title("Ustawienia Gry")

    welcome_label = tk.Label(root, text="Witaj w grze!")
    instructions_label = tk.Label(root, text="Wybierz wariant gry i wpisz swoje imię (opcjonalnie).")
    welcome_label.pack()
    instructions_label.pack()

    main_frame = tk.Frame(root)  # Tworzenie głównej ramki

    name_frame = tk.Frame(main_frame)
    name_label = tk.Label(name_frame, text="Wpisz swoje imię:")
    name_entry = tk.Entry(name_frame)
    name_label.pack()
    name_entry.pack()
    name_frame.pack()

    game_variant_frame = tk.Frame(main_frame)

    game_variants = tk.Label(game_variant_frame, text="Wybierz rozgrywkę:")
    game1_button = tk.Button(game_variant_frame, text="Gra 1", command=game1_button_callback)
    game2_button = tk.Button(game_variant_frame, text="Gra 2", command=game2_button_callback)
    game_variants.pack()
    game1_button.pack()
    game2_button.pack()
    game1_frame = tk.Frame(game_variant_frame)
    game1_frame.pack()
    game2_frame = tk.Frame(game_variant_frame)
    game2_frame.pack()
    game_variant_frame.pack()

    main_frame.pack()

    confirm_set = tk.Label(root, text="Jeśli jesteś gotowy to zatwierdź ustawienia:")
    confirm_button = tk.Button(root, text="Zatwierdź", command=confirm_button_callback)
    confirm_button.pack(side=tk.BOTTOM)
    confirm_set.pack(side=tk.BOTTOM)

    root.mainloop()  # Uruchomienie pętli głównego okna

    if name_entry.get():
        player_name = name_entry.get()
    else:
        player_name = 'Player'

    return player_name


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


def show_answers(player_name, answers=None, correct_count=None, incorrect_count=None, total_time = None):
    # Tworzenie pustego obrazu o kolorze błękitnym
    main_image = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
    main_image[:, :] = WHITE

    if total_time is None and answers is not None:
        answers_to_display = []

        for idx, a in enumerate(answers):
            face, image_answer = answers[idx]
            ready_answer = prepare_answer(face, image_answer)
            answers_to_display.append(ready_answer)

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
            cv2.putText(main_image, text, org=(380, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=BLACK, thickness=2)
        else:
            raise ValueError
    else:
        if total_time is None:
            raise ValueError
        minutes = int(total_time // 60)
        total_time %= 60
        text = f"SUMMARY for {player_name}: total time: {minutes} min {total_time:.2f} sec."
        cv2.putText(main_image, text, org=(380, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=BLACK, thickness=2)

    # Wyświetlanie obrazu głównego
    cv2.imshow('Results', main_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    global game_variant, difficult_level
    # settings
    player_name = set_the_game()
    model = Res10()
    if game_variant == 1:
        answers, correct_count, incorrect_count = play_1(model, difficult=difficult_level)
        show_answers(player_name, answers=answers, correct_count=correct_count, incorrect_count=incorrect_count)
        try:
            show_answers(player_name, answers=answers, correct_count=correct_count, incorrect_count=incorrect_count)
        except ValueError:
            print('Gra została przerwana')
    elif game_variant == 2:
        total_time = play_2(model)
        try:
            show_answers(player_name, total_time=total_time)
        except ValueError:
            print('Gra została przerwana')


if __name__ == '__main__':
    main()
