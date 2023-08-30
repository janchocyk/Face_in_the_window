# Face Detection Game with OpenCV

This is a face detection game application with two different game variants. The game uses various face detection models to detect faces in real-time video frames and challenges the player to interact with the game using their face.

## Table of Contents

- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
- [Game Variants](#game-variants)
- [Models](#models)
- [Technologies](#technologies)
- [License](#license)

## Installation

1. Clone the repository:

   ```git clone https://github.com/your-username/face-detection-game.git```

2. Install the required librarys:

   ```pip install -r requirements.txt```

## Usage

Run the `application\main.py` script to start the game. The game will prompt you to select a game variant, set the difficulty level, and optionally enter your name. Follow the on-screen instructions and interact with the game using your face to complete the challenges.
After completing the game, a summary will be displayed showing the number of correct and incorrect detections.

## Game Variants

### Game Variant 1

In this variant, rectangles will appear on the screen that will change their position. Your task is to position your face in the video preview so that it is inside the rectangle. The difficulty level determines how often the rectangles change. A point is awarded if the program recognizes a face inside the frame.

### Game Variant 2

In this variant, you will see randomly generated rectangles on the screen. You have to move your face and set it inside them. If the program detects a face inside the rectangle, it will disappear. Your goal is to make all the rectangles disappear in the shortest possible time.

## Models

The game uses different face detection models for each game variant:

1. **Res10 Model:** A pre-trained deep learning model based on the Single Shot MultiBox Detector (SSD) framework. It can quickly detect faces in images.

2. **YOLOv8 Model:** The YOLO (You Only Look Once) model that can detect objects, including faces, in real-time. The Ultralytics implementation is used in this game.

3. **Cascade Classifier:** A traditional computer vision model based on Haar-like features that can detect faces efficiently.

4. **Custom Model:** A custom deep learning model trained to classify faces. It determines if an image contains a face or not.

You can choose the model you want to use in the main() function of the main.py script.
You can also implement your own model in models.py. Do it with a class where the constructor will be the initialization of the model and the "detect" method will detect the image. It should return a list with a tuple containing the coordinates of the corners of the rectangle surrounding the face and an image with a drawn rectangle in case of correct detection or an empty image in case of incorrect detection.

## Technologies

The following technologies were used in this project:

- OpenCV: A computer vision library used for image and video processing tasks.
- NumPy: A library for numerical computations in Python.
- TensorFlow: An open-source machine learning framework used for deep learning models.
- Ultralytics: A library that includes pre-trained models for object detection tasks.
- Haarcascades: Pre-trained models for the Cascade Classifier face detection method.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

