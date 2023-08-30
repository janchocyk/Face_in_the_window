import cv2
import numpy as np
import tkinter as tk

from models import Res10, CascadeClassifier, YOLOv8, Own_model
from games import play_1, play_2, SCREEN_HEIGHT, SCREEN_WIDTH, RECT_HEIGHT, RECT_WIDTH, RED, BLACK, WHITE

def set_the_game() -> tuple[str, int, int]:
    """
    Initial game settings.

    Returns:
        Tuple[str, int, int]: A tuple containing player's name, game variant, and difficulty level.
    """
    game_variant = 1
    difficulty_level = 1

    def clear_frame(frame):
        for widget in frame.winfo_children():
            widget.destroy()

    def create_game1_frame():
        game2_frame.pack_forget()
        game1_frame.pack()
        difficulty1_level_label = tk.Label(game1_frame, text="Set difficulty level:")
        # Creating three radio buttons
        radio1_button1 = tk.Radiobutton(game1_frame, text="Easy", variable=selected_option1, value=1, command=update_difficulty1)
        radio1_button2 = tk.Radiobutton(game1_frame, text="Medium", variable=selected_option1, value=2, command=update_difficulty1)
        radio1_button3 = tk.Radiobutton(game1_frame, text="Hard", variable=selected_option1, value=3, command=update_difficulty1)
        selected_option1.set(None)
        difficulty1_level_label.pack()
        radio1_button1.pack()
        radio1_button2.pack()
        radio1_button3.pack()

    def create_game2_frame():
        game1_frame.pack_forget()
        game2_frame.pack()
        difficulty2_level_label = tk.Label(game2_frame, text="Set frame size:")
        # Creating three radio buttons
        radio2_button1 = tk.Radiobutton(game2_frame, text="Normal", variable=selected_option2, value=1, command=update_difficulty2)
        radio2_button2 = tk.Radiobutton(game2_frame, text="Small", variable=selected_option2, value=2, command=update_difficulty2)
        selected_option2.set(None)
        difficulty2_level_label.pack()
        radio2_button1.pack()
        radio2_button2.pack()

    def game1_button_callback():
        nonlocal game_variant
        game_variant = 1
        game1_button.config(state=tk.DISABLED)
        game2_button.config(state=tk.NORMAL)
        clear_frame(game2_frame)
        create_game1_frame()

    def game2_button_callback():
        nonlocal game_variant
        game_variant = 2
        game2_button.config(state=tk.DISABLED)
        game1_button.config(state=tk.NORMAL)
        clear_frame(game1_frame)
        create_game2_frame()

    def update_difficulty1():
        nonlocal difficulty_level
        difficulty_level = selected_option1.get()

    def update_difficulty2():
        nonlocal difficulty_level
        difficulty_level = selected_option2.get()

    def confirm_button_callback():
        root.quit()

    root = tk.Tk()  # Creating the main window
    selected_option1 = tk.StringVar(value=None)
    selected_option2 = tk.StringVar(value=None)

    root.title("Game Settings")

    welcome_label = tk.Label(root, text="Welcome to the game!")
    instructions_label = tk.Label(root, text="Choose the game variant and enter your name (optional).")
    welcome_label.pack()
    instructions_label.pack()

    main_frame = tk.Frame(root)  # Creating the main frame

    name_frame = tk.Frame(main_frame)
    name_label = tk.Label(name_frame, text="Enter your name:")
    name_entry = tk.Entry(name_frame)
    name_label.pack()
    name_entry.pack()
    name_frame.pack()

    game_variant_frame = tk.Frame(main_frame)

    game_variants = tk.Label(game_variant_frame, text="Choose the game mode:")
    game1_button = tk.Button(game_variant_frame, text="Game 1", command=game1_button_callback)
    game2_button = tk.Button(game_variant_frame, text="Game 2", command=game2_button_callback)
    game_variants.pack()
    game1_button.pack()
    game2_button.pack()
    game1_frame = tk.Frame(game_variant_frame)
    game1_frame.pack()
    game2_frame = tk.Frame(game_variant_frame)
    game2_frame.pack()
    game_variant_frame.pack()

    main_frame.pack()

    confirm_set = tk.Label(root, text="If you're ready, confirm the settings:")
    confirm_button = tk.Button(root, text="Confirm", command=confirm_button_callback)
    confirm_button.pack(side=tk.BOTTOM)
    confirm_set.pack(side=tk.BOTTOM)

    root.mainloop()  # Start the main window loop

    if name_entry.get():
        player_name = name_entry.get()
    else:
        player_name = 'Player'

    return player_name, game_variant, difficulty_level


def prepare_answer(faces: list[tuple[int, int, int, int]], image_answer: np.ndarray) -> np.ndarray:
    """
    Prepares a visual answer based on game results.

    Args:
        faces (List[Tuple[int, int, int, int]]): List of face bounding box coordinates.
        image_answer (np.ndarray): Image to modify.

    Returns:
        np.ndarray: Processed image with answer indication.
    """
    if len(faces) == 0:
        x, y, w, h = (0, 0, RECT_WIDTH, RECT_HEIGHT)
        answer = cv2.rectangle(image_answer, (x, y), (x + w, y + h), RED, 8)
        text = "Incorrect"
        ready_answer = cv2.putText(image_answer, text, org=(30, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=RED, thickness=2)
    else:
        text = "Correct"
        ready_answer = cv2.putText(image_answer, text, org=(30, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=BLACK, thickness=2)
    return ready_answer


def show_answers(player_name: str, answers: list[np.ndarray]=None, correct_count: int=None, incorrect_count: int=None, total_time: float= None) -> None:
    """
    Displays game results or summary with information on the image.

    Args:
        player_name (str): Player's name.
        answers (List[np.ndarray]: List of answers in the form of cropped images. Defaults to None.
        correct_count (int, optional): Number of correct answers. Defaults to None.
        incorrect_count (int, optional): Number of incorrect answers. Defaults to None.
        total_time (float, optional): Total game time. Defaults to None.
    """
    # Creating an empty blue-colored image
    main_image = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
    main_image[:, :] = WHITE

    if total_time is None and answers is not None:
        answers_to_display = []

        for idx, a in enumerate(answers):
            face, image_answer = answers[idx]
            ready_answer = prepare_answer(face, image_answer)
            answers_to_display.append(ready_answer)

        # Number of images in a row and column
        num_rows = 2
        num_columns = 5

        # Space between images
        spacing = 10

        # Check if the number of images to display matches the number of cells
        if len(answers_to_display) == num_rows * num_columns:
            for row in range(num_rows):
                for col in range(num_columns):
                    # Coordinates of the top-left corner of the image on the main image
                    x = 115 + col * (RECT_WIDTH + spacing)
                    y = 180 + row * (RECT_HEIGHT + spacing)

                    # Get the image from the 'answers_to_display' array
                    image = answers_to_display[row * num_columns + col]

                    # Resize the image to 250x200 (if different)
                    if image.shape[:2] != (RECT_HEIGHT, RECT_WIDTH):
                        image = cv2.resize(image, (RECT_WIDTH, RECT_HEIGHT))

                    # Place the image on the main image
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

    # Display the main image
    cv2.imshow('Results', main_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    """
    The main game function.
    """
    # settings
    player_name, game_variant, difficulty_level = set_the_game()
    model = Res10()
    if game_variant == 1:
        answers, correct_count, incorrect_count = play_1(model, difficulty=difficulty_level)
        try:
            show_answers(player_name, answers=answers, correct_count=correct_count, incorrect_count=incorrect_count)
        except ValueError:
            print('The game has been interrupted.')
    elif game_variant == 2:
        total_time = play_2(model)
        try:
            show_answers(player_name, total_time=total_time)
        except ValueError:
            print('The game has been interrupted.')


if __name__ == '__main__':
    main()

