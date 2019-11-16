import pandas as pd
import numpy as np
import cv2 as cv

data_path = '../data/fer2013/fer2013.csv'
img_size = (48, 48)

emotions_dict = {1: "angry", 2: "disgust", 3: "scared", 4: "happy", 5: "sad", 6: "surprised", 7: "neutral"}


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
    return x_train, y_train, x_val, y_val, x_test, y_test


x_train, y_train, x_val, y_val, x_test, y_test = load_data(data_path)
