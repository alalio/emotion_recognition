import imutils
import cv2
import numpy as np
from src.models.cnn import model_1, model_2, model_ref, model_1_2
from keras.preprocessing.image import img_to_array

detection_model_path = '../cascades/haarcascades/haarcascade_frontalface_default.xml'

face_detection = cv2.CascadeClassifier(detection_model_path)
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

# load the model
model_path = '../63_accu_model_2.ckpt'
model = model_2()
model.load_weights(model_path)

cv2.namedWindow('your_face')
camera = cv2.VideoCapture(0)
while True:
    frame = camera.read()[1]
    frame = imutils.resize(frame, width=300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect face
    faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48), flags=cv2.CASCADE_SCALE_IMAGE)

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
        #face = face.astype("float") / 255.0
        face = img_to_array(face)
        face = np.expand_dims(face, axis=0)

        # predict on the model
        prediction = model.predict(face)[0]
        label = EMOTIONS[prediction.argmax()]

    else:
        continue

    for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, prediction)):
        text = "{}: {:.2f}%".format(emotion, prob * 100)

        w = int(prob * 300)
        # histogram
        cv2.rectangle(canvas, (7, (i * 35) + 5), (w, (i * 35) + 35), (0, 0, 255), -1)
        cv2.putText(canvas, text, (10, (i * 35) + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
        # rectangle arround head
        cv2.putText(frameClone, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)

    cv2.imshow('your_face', frameClone)
    cv2.imshow("Probabilities", canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
