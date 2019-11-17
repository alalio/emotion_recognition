from src.load_data import load_data
from src.models.cnn import model_1
from keras.callbacks import ModelCheckpoint


continue_training = True


data_path = '../data/fer2013/fer2013.csv'
img_size = (48, 48)

x_train, y_train, x_val, y_val, x_test, y_test = load_data(data_path)

model = model_1()
model.summary()
save_path = './save_model.ckpt'
save_model_callback = ModelCheckpoint(filepath=save_path, verbose=1)
if continue_training:
    model.load_weights(save_path)

model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=1000, batch_size=100, callbacks=[save_model_callback])
    

