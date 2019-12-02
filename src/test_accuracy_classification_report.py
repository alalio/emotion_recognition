from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import pickle
from keras.models import load_model
import os
from src.load_data import load_data
import numpy as np

# Drive
drive_data_path = '/content/drive/My Drive/abdou/emotion_detection/fer2013.csv'

# Local
local_data_path = '../data/fer2013/fer2013.csv'
local_data_path2 = 'data/fer2013/fer2013.csv'

data_path_r = None
if os.path.isfile(local_data_path2):
    data_path_r = local_data_path2
elif os.path.isfile(local_data_path):
    data_path_r = local_data_path
else:
    data_path_r = drive_data_path

img_size = (48, 48)

x_train, y_train, x_val, y_val, x_test, y_test = load_data(data_path_r)

history = pickle.load(open("history.bin", "rb"))

save_path_2 = "save_model_2.ckpt"
model = load_model(save_path_2)

pred_y = model.predict(x_test)

print(pred_y[0], y_test[0])


def make_int_y(y):
    y = np.array(y)
    y__ = []
    for k in y:
        j = np.argmax(k)
        l = [0., 0., 0., 0., 0., 0., 0.]
        l[j] = 1.
        y__.append(l)
    return np.array(y__)


pred_y = make_int_y(pred_y)
print(pred_y[0], y_test[0])

print(classification_report(pred_y.argmax(axis=1), y_test.argmax(axis=1)))
print(confusion_matrix(pred_y.argmax(axis=1), y_test.argmax(axis=1)))
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
