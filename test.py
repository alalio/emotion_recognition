import imutils
import cv2
import numpy as np
from src.models.cnn import model_1
from keras.preprocessing.image import img_to_array

import matplotlib.pyplot as plt


detection_model_path = './cascades/haarcascades/haarcascade_frontalface_default.xml'

face_detection = cv2.CascadeClassifier(detection_model_path)
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

# load the model
model_path = './60_accu_model.ckpt'
model = model_1()
model.load_weights(model_path)

cv2.namedWindow('your_face')
camera = cv2.VideoCapture(0)

frame = camera.read()[1]
frame = imutils.resize(frame, width=300)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# detect face
faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                        flags=cv2.CASCADE_SCALE_IMAGE)

canvas = np.zeros((250, 300, 3), dtype="uint8")
frameClone = frame.copy()
if len(faces) > 0:
    faces = sorted(faces, reverse=True,
                   key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
    # detect face pos
    (fX, fY, fW, fH) = faces

    # extract face and reshape it to (48, 48, 1)
    face = gray[fY:fY + fH, fX:fX + fW]
    face = cv2.resize(face, (48, 48))
    face = face.astype("float") / 255.0
    face = img_to_array(face)
    face = np.expand_dims(face, axis=0)

    img = face.reshape((48, 48))

    # predict on the model
    prediction = model.predict(face)[0]
    label = EMOTIONS[prediction.argmax()]

    imgplot = plt.imshow(img)
    plt.show()
    print(label)
