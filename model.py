from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D


class Model:
    # Set the expected shape of input here.
    input_shape = (1, 28, 28)

    def create(self):
        # Define model.
        model = Sequential()

        model.add(ZeroPadding2D((1, 1), input_shape=Model.input_shape))
        model.add(Conv2D(32, (5, 5), activation='relu'))
        model.add(Conv2D(32, (5, 5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(10, activation='softmax'))

        # Compile model.
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model
