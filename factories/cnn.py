import random
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
import numpy as np
from model_factory import ModelFactory


class EmptyError(Exception):
    pass


class Factory(ModelFactory):
    def __init__(self):
        super().__init__()

        # Set the expected shape of input here.
        # TensorFlow ordering. Reverse for use with Theano.
        self.input_shape = (1, 28, 28)
        self.n_classes = None
        self.batch_size = 1
        self.training_samples = 0
        self.validation_samples = 0

    def create(self):
        if (self.n_classes is None or self.n_classes == 0):
            raise EmptyError('The number of classifications must be given.')

        # Define model.
        model = Sequential()

        model.add(
            Conv2D(
                32,
                (5, 5),
                activation='relu',
                input_shape=self.input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.n_classes, activation='softmax'))

        # Compile model.
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def load_data(self):
        (
            (training_features, training_labels),
            (validation_features, validation_labels)) = mnist.load_data()

        training_features = self.process_inputs(training_features)
        training_labels = self.process_labels(training_labels)
        self.training_samples = len(training_features)

        validation_features = self.process_inputs(validation_features)
        validation_labels = self.process_labels(validation_labels)
        self.validation_samples = len(validation_features)

        return (
            (training_features, training_labels),
            (validation_features, validation_labels))

    def get_generators(self):
        (
            (training_features, training_labels),
            (validation_features, validation_labels)) = self.load_data()

        training_generator = self.create_generator(
            training_features,
            training_labels,
            self.batch_size)

        validation_generator = self.create_generator(
            validation_features,
            validation_labels,
            self.batch_size)

        return training_generator, validation_generator

    def create_generator(self, features, labels, batch_size):
        batch_features = np.zeros((batch_size,) + self.input_shape)
        batch_labels = np.zeros((batch_size, labels.shape[1]))

        while True:
            for i in range(batch_size):
                index = random.randrange(len(features))
                batch_features[i] = features[index]
                batch_labels[i] = labels[index]
            yield batch_features, batch_labels

    def process_inputs(self, features):
        # Reshape training data for input into the model.
        new_shape = (features.shape[0],) + self.input_shape
        features = np.reshape(features, new_shape)

        # Normalize input.
        features = features / 255

        return features

    def process_labels(self, labels):
        # Create one hot encodings.
        labels = np_utils.to_categorical(labels)
        self.n_classes = labels.shape[1]

        return labels
