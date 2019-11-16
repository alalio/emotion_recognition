from src.load_data import load_data
from src.models.cnn import model_1


data_path = '../data/fer2013/fer2013.csv'
img_size = (48, 48)

x_train, y_train, x_val, y_val, x_test, y_test = load_data(data_path)

model = model_1()
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, batch_size=50)


