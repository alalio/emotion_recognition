from src.load_data import load_data
from src.models.cnn import model_1
from keras.callbacks import ModelCheckpoint
import os
import pickle

continue_training = True

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

model = model_1()
model.summary()
save_path = 'save_model.ckpt'
save_model_callback = ModelCheckpoint(filepath=save_path, verbose=1)
if continue_training and os.path.isfile(save_path):
    model.load_weights(save_path)

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50,
                    batch_size=100, callbacks=[save_model_callback])

pickle.dump(history, open("history.bin", "wb"))

save_path_2 = "save_model_2.ckpt"
model.save(save_path_2)
