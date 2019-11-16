import pandas as pd
import numpy as np
from operator import methodcaller
from keras.utils.np_utils import to_categorical

data_path = '/content/drive/My Drive/abdou/emotion_detection/fer2013.csv'
img_size = (48, 48)

emotions_dict = {1: "angry", 2: "disgust", 3: "scared", 4: "happy", 5: "sad", 6: "surprised", 7: "neutral"}

def preprocess_data(data):
  for i in range(len(data)):
    data[i] = list(map(int, data[i]))
    data[i] = np.array(data[i])
    data[i] = data[i].reshape((48,48,1))
  return np.array(data)


def load_data(data_path, train_val_split=0.8):
    data = pd.read_csv(data_path)
    data_train = data[data['Usage'] == 'Training']
    data_test = data[data['Usage'] != 'Training']

    x_test = data_test['pixels']
    y_test = data_test['emotion']

    x_train_not_split = data_train['pixels']
    y_train_not_split = data_train['emotion']

    splitter = np.random.rand(len(x_train_not_split)) < train_val_split
    x_train = x_train_not_split[splitter]
    y_train = y_train_not_split[splitter]

    x_val = x_train_not_split[~splitter]
    y_val = y_train_not_split[~splitter]

    x_train, y_train, x_val, y_val, x_test, y_test = x_train.values, y_train.values, x_val.values, y_val.values, x_test.values, y_test.values

    x_train = list(map(methodcaller("split", " "), x_train))
    x_train = preprocess_data(x_train)

    x_val = list(map(methodcaller("split", " "), x_val))
    x_val = preprocess_data(x_val)

    x_test = list(map(methodcaller("split", " "), x_test))
    x_test = preprocess_data(x_test)

    y_train = to_categorical(np.asarray(y_train), 7)
    y_val = to_categorical(np.asarray(y_val), 7)
    y_test = to_categorical(np.asarray(y_test), 7)

    return x_train, y_train, x_val, y_val, x_test, y_test


x_train, y_train, x_val, y_val, x_test, y_test = load_data(data_path)
