from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from keras.models import Sequential

def model_1(input_shape=((48, 48, 1)), num_classes=7):
    """
    VGG face inspired
    https://github.com/rcmalli/keras-vggface/blob/master/keras_vggface/models.py

    :return: model
    """
    model = Sequential()

    # Block 1
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool1'))

    # Block 2
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool2'))

    # Block 3
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool3'))

    # Block 4
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool4'))

    # Block 5
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool5'))

    # Classification block
    model.add(Flatten(name='flatten'))
    model.add(Dense(4096, name='fc6'))
    model.add(Activation('relu', name='fc6/relu'))
    model.add(Dense(4096, name='fc7'))
    model.add(Activation('relu', name='fc7/relu'))
    model.add(Dense(num_classes, name='fc8'))
    model.add(Activation('softmax', name='fc8/softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

