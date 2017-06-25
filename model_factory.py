from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.utils import np_utils


class ModelFactory:
    def __init__(self):
        # Set the expected shape of input here.
        self.input_shape = (1, 28, 28)
        self.n_classes = None

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
        model.add(Dense(self.n_classes, activation='softmax'))

        # Compile model.
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model

    def load_data(self):
        return mnist.load_data()

    def process_inputs(self, X):
        # Reshape training data for input into the model.
        X_shape = (X.shape[0],) + self.input_shape
        X = X.reshape(X_shape)

        # Normalize inputs.
        X = X / 255

        return X

    def process_labels(self, Y):
        # Create one hot encodings.
        Y = np_utils.to_categorical(Y)
        self.n_classes = Y.shape[1]

        return Y
