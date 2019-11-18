from keras.models import Sequential, Model
from keras.layers import SeparableConv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, GlobalMaxPooling2D, BatchNormalization, Activation, Layer, Input, Conv2D, add, GlobalAveragePooling2D
from keras.regularizers import l2


def model_1(input_shape=((48, 48, 1)), num_classes=7):
    """
    VGG face inspired
    https://github.com/rcmalli/keras-vggface/blob/master/keras_vggface/models.py

    :return: model
    """
    model = Sequential()
    reg = l2(0.01)

    # Block 1
    model.add(SeparableConv2D(16, (3, 3), padding='same', kernel_regularizer=reg, name='conv1_1', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(SeparableConv2D(16, (3, 3), padding='same', kernel_regularizer=reg, name='conv1_2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool1'))
    model.add(Dropout(0.2))

    # Block 2
    model.add(SeparableConv2D(32, (3, 3), padding='same', kernel_regularizer=reg, name='conv2_1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(SeparableConv2D(32, (3, 3), padding='same', kernel_regularizer=reg, name='conv2_2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool2'))
    model.add(Dropout(0.2))

    # Block 3
    model.add(SeparableConv2D(32, (3, 3), padding='same', kernel_regularizer=reg, name='conv3_1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(SeparableConv2D(32, (3, 3), padding='same', kernel_regularizer=reg, name='conv3_2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(SeparableConv2D(32, (3, 3), padding='same', kernel_regularizer=reg, name='conv3_3'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((3, 3), strides=(2, 2), name='pool3'))
    model.add(Dropout(0.2))

    # Block 4
    model.add(SeparableConv2D(64, (3, 3), padding='same', kernel_regularizer=reg, name='conv4_1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(SeparableConv2D(64, (3, 3), padding='same', kernel_regularizer=reg, name='conv4_2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(SeparableConv2D(64, (3, 3), padding='same', kernel_regularizer=reg, name='conv4_3'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool4'))
    model.add(Dropout(0.2))

    # Classification block
    model.add(Flatten(name='flatten'))
    model.add(Dense(64, name='fc6'))
    model.add(Activation('relu', name='fc6/relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, name='fc7'))
    model.add(Activation('relu', name='fc7/relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, name='fc8'))
    model.add(Activation('softmax', name='fc8/softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def model_1_2(input_shape=((48, 48, 1)), num_classes=7):
    """
    VGG face inspired
    https://github.com/rcmalli/keras-vggface/blob/master/keras_vggface/models.py

    :return: model
    """
    model = Sequential()
    reg = l2(0.01)

    # Block 1
    model.add(Conv2D(16, (3, 3), padding='same', kernel_regularizer=reg, name='conv1_1', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(16, (3, 3), padding='same', kernel_regularizer=reg, name='conv1_2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool1'))
    model.add(Dropout(0.2))

    # Block 2
    model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=reg, name='conv2_1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=reg, name='conv2_2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool2'))
    model.add(Dropout(0.2))

    # Block 3
    model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=reg, name='conv3_1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=reg, name='conv3_2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=reg, name='conv3_3'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((3, 3), strides=(2, 2), name='pool3'))
    model.add(Dropout(0.2))

    # Block 4
    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=reg, name='conv4_1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=reg, name='conv4_2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=reg, name='conv4_3'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool4'))
    model.add(Dropout(0.2))

    # Classification block
    model.add(Flatten(name='flatten'))
    model.add(Dense(64, name='fc6'))
    model.add(Activation('relu', name='fc6/relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, name='fc7'))
    model.add(Activation('relu', name='fc7/relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, name='fc8'))
    model.add(Activation('softmax', name='fc8/softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def model_2(input_shape=((48, 48, 1)), num_classes=7):
    """
    VGG face inspired
    https://github.com/rcmalli/keras-vggface/blob/master/keras_vggface/models.py

    :return: model
    """
    model = Sequential()
    reg = l2(0.01)

    # Block 1
    model.add(SeparableConv2D(32, (3, 3), padding='same', kernel_regularizer=reg, name='conv1_1', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(SeparableConv2D(32, (3, 3), padding='same', kernel_regularizer=reg, name='conv1_2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool1'))
    model.add(Dropout(0.2))

    # Block 2
    model.add(SeparableConv2D(64, (3, 3), padding='same', kernel_regularizer=reg, name='conv2_1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(SeparableConv2D(64, (3, 3), padding='same', kernel_regularizer=reg, name='conv2_2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool2'))
    model.add(Dropout(0.2))

    # Block 3
    model.add(SeparableConv2D(64, (3, 3), padding='same', kernel_regularizer=reg, name='conv3_1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(SeparableConv2D(64, (3, 3), padding='same', kernel_regularizer=reg, name='conv3_2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(SeparableConv2D(64, (3, 3), padding='same', kernel_regularizer=reg, name='conv3_3'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((3, 3), strides=(2, 2), name='pool3'))
    model.add(Dropout(0.2))

    # Block 4
    model.add(SeparableConv2D(128, (3, 3), padding='same', kernel_regularizer=reg, name='conv4_1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(SeparableConv2D(128, (3, 3), padding='same', kernel_regularizer=reg, name='conv4_2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(SeparableConv2D(128, (3, 3), padding='same', kernel_regularizer=reg, name='conv4_3'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool4'))
    model.add(Dropout(0.2))

    # Classification block
    model.add(Flatten(name='flatten'))
    model.add(Dense(64, name='fc6'))
    model.add(Activation('relu', name='fc6/relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, name='fc7'))
    model.add(Activation('relu', name='fc7/relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, name='fc8'))
    model.add(Activation('softmax', name='fc8/softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def model_ref():
    reg = l2(0.01)

    # base
    img_input = Input((48,48,1))
    x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=reg)(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=reg)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # module 1
    residual = Conv2D(16, (1, 1), strides=(2, 2), padding='same')(x)
    residual = BatchNormalization()(residual)

    x = Conv2D(16, (3, 3), padding='same', kernel_regularizer=reg)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(16, (3, 3), padding='same', kernel_regularizer=reg)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = add([x, residual])

    # module 2
    residual = Conv2D(32, (1, 1), strides=(2, 2), padding='same')(x)
    residual = BatchNormalization()(residual)

    x = Conv2D(32, (3, 3), padding='same', kernel_regularizer=reg)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(32, (3, 3), padding='same', kernel_regularizer=reg)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = add([x, residual])

    # module 3
    residual = Conv2D(64, (1, 1), strides=(2, 2), padding='same')(x)
    residual = BatchNormalization()(residual)

    x = Conv2D(64, (3, 3), padding='same', kernel_regularizer=reg)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same', kernel_regularizer=reg)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = add([x, residual])

    # module 4
    residual = Conv2D(128, (1, 1), strides=(2, 2), padding='same')(x)
    residual = BatchNormalization()(residual)

    x = Conv2D(128, (3, 3), padding='same', kernel_regularizer=reg)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), padding='same', kernel_regularizer=reg)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = add([x, residual])

    x = Conv2D(7, (3, 3), kernel_regularizer=reg, padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    output = Activation('softmax',name='predictions')(x)

    model = Model(img_input, output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
