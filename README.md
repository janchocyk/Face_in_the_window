# Face Detection Game with OpenCV

This is a Python script that implements a simple face detection game using OpenCV. The game displays rectangles on a live video stream where players need to identify whether a face is present within the rectangle. The game keeps track of correct and incorrect detections and provides a summary of the results at the end.

## Features

- Uses OpenCV for real-time video processing and display.
- Implements two different face detection models: Res10 and CascadeClassifier.
- Tracks correct and incorrect detections while playing the game.
- Displays rectangles on the video stream for the player to identify faces.
- Generates a summary with the number of correct and incorrect detections after the game.

## Getting Started

1. Clone this repository (on Command Prompt or Bash):
   
```git clone <repository_url>```

```cd <repository_name>```
   
3. Install requirement librarys (on Command Prompt or Bash):

```pip install -r requirements.txt```

5. Run the script:

```python main.py```
   
7. Play the game:
- The game will display rectangles on the video stream. You need to identify whether a face is present within each rectangle.
- Press 'q' to exit the game once you're done.
5. View the summary:
- After completing the game, a summary will be displayed showing the number of correct and incorrect detections.

## Models
The game supports two face detection models:

1. Res10 Model: This model uses a pre-trained deep learning model to detect faces - it is defoult model.
2. Cascade Classifier Model: This model uses a Haar Cascade classifier to detect faces.

You can choose the model you want to use in the main() function of the main.py script.
You can also implement your own model in models.py. Do it with a class where the constructor will be the initialization of the model and the "detection" method will detect the image. It should return a list with a tuple containing the coordinates of the corners of the rectangle surrounding the face and an image with a drawn rectangle in case of correct detection or an empty image in case of incorrect detection.

## Acknowledgments
This project was inspired by the desire to create an interactive game to practice face detection using OpenCV. It's a great way to learn about real-time video processing and explore different face detection techniques.

Feel free to contribute to the project by improving the user interface, adding more features, or optimizing the code. Happy face detecting!





