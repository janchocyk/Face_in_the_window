import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from typing import List, Tuple

GREEN = (0, 255, 0)

class Res10:
    def __init__(self, confidence: float = 0.7):
        """
        Initializes the Res10 face detection model.

        Args:
            confidence (float, optional): Confidence threshold. Defaults to 0.7.
        """
        self.model = cv2.dnn.readNetFromCaffe("ModelRes10\\deploy.prototxt.txt", "ModelRes10\\res10_300x300_ssd_iter_140000.caffemodel")
        self.confidence = confidence

    def detect(self, image: np.ndarray) -> tuple[list[tuple[int, int, int, int]], np.ndarray]:
        """
        Detects faces in the input image using the Res10 model.

        Args:
            image (np.ndarray): Input image.

        Returns:
            Tuple[List[Tuple[int, int, int, int]], np.ndarray]: Detected faces and modified image.
        """
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.model.setInput(blob)
        detections = self.model.forward()
        faces = []
        confidences = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.confidence:
                box = detections[0, 0, i, 3:7] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
                (startX, startY, endX, endY) = box.astype("int")
                faces.append((startX, startY, endX, endY))
                confidences.append(confidence)
        if faces:
            max_confidence = max(confidences)
            best_face_idx = confidences.index(max_confidence)
            best_face = faces[best_face_idx]
            faces = [best_face]
            startX, startY, endX, endY = best_face
            cv2.rectangle(image, (startX, startY), (endX, endY), GREEN, 3)
        return faces, image

class CascadeClassifier:
    def __init__(self):
        """
        Initializes the Cascade Classifier model for face detection.
        """
        self.model = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Detects faces in the input image using the Cascade Classifier model.

        Args:
            image (np.ndarray): Input image.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Detected faces and modified image.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.model.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) != 0:
            startX, startY, endX, endY = faces[0]
            cv2.rectangle(image, (startX, startY), (endX, endY), GREEN, 3)
        return faces, image

class YOLOv8:
    def __init__(self):
        """
        Initializes the YOLOv8 model for face detection.
        """
        self.model = YOLO('YOLO_v8/best.pt')

    def detect(self, image: np.ndarray) -> tuple[list[tuple[int, int, int, int]], np.ndarray]:
        """
        Detects faces in the input image using the YOLOv8 model.

        Args:
            image (np.ndarray): Input image.

        Returns:
            Tuple[List[Tuple[int, int, int, int]], np.ndarray]: Detected faces and modified image.
        """
        results = self.model(image, conf=0.3)
        boxes = results[0].boxes
        faces = []
        for box in boxes:
            startX = int(box.xyxy.tolist()[0][0])
            startY = int(box.xyxy.tolist()[0][1])
            endX = int(box.xyxy.tolist()[0][2])
            endY = int(box.xyxy.tolist()[0][3])
            cv2.rectangle(image, (startX, startY), (endX, endY), GREEN, 3)
            faces = [(startX, startY, endX, endY)]
        return faces, image

class Own_model:
    def __init__(self):
        """
        Initializes the custom model for face classification.
        """
        self.model = tf.keras.models.load_model('Own_model')

    def detect(self, image: np.ndarray) -> tuple[list[tuple[int, int, int, int]], np.ndarray]:
        """
        Detects faces in the input image using the custom model.

        Args:
            image (np.ndarray): Input image.

        Returns:
            Tuple[List[Tuple[int, int, int, int]], np.ndarray]: Detected faces and modified image.
        """
        classes = ['face', 'not_face']
        image_array = cv2.resize(image, (224, 224))
        image_array = tf.expand_dims(image_array, 0)
        predictions = self.model.predict(image_array)
        predicted_class_index = tf.argmax(predictions[0])
        predicted_class_name = classes[predicted_class_index]
        if predicted_class_name == 'face':
            (startX, startY, endX, endY) = (5, 5, 195, 245)
            cv2.rectangle(image, (startX, startY), (endX, endY), GREEN, 3)
            faces = [(startX, startY, endX, endY)]
        else:
            faces = []
        return faces, image
