import cv2
import numpy as np

# # Ścieżka do pliku z modelem prototxt
# prototxt_path = "ModelRes10/deploy.prototxt.txt"
# # Ścieżka do wagi modelu
# model_path = "ModelRes10/res10_300x300_ssd_iter_140000.caffemodel"

class Res10():

    def __init__(self):
        self.model = cv2.dnn.readNetFromCaffe("ModelRes10/deploy.prototxt.txt", "ModelRes10/res10_300x300_ssd_iter_140000.caffemodel")

    def detection(self, image):
        # Przygotowanie obrazu do analizy (normalizacja, skalowanie)
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

        # Przekazanie obrazu przez sieć neuronową
        self.model.setInput(blob)
        detections = self.model.forward()
        faces = []
        confidences = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Sprawdzanie, czy pewność jest wystarczająco wysoka
            if confidence > 0.6:
                # Pobieranie współrzędnych bounding box
                box = detections[0, 0, i, 3:7] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
                (startX, startY, endX, endY) = box.astype("int")
                # Narysowanie prostokątu wokół twarzy
                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 3)
                confidences.append(confidence)
                if confidence == max(confidences):
                    face = (startX, startY, endX, endY)
                    faces.insert(0, face)
        return faces, image


class CascadeClassifier():

    def __init__(self):
        self.model = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detection(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.model.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) != 0:
            startX, startY, endX, endY = faces
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 3)

        return faces, image