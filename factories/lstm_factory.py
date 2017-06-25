from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils
import numpy as np

from model_factory import ModelFactory


class LSTMFactory(ModelFactory):
    def __init__(self):
        self.input_shape = None
        self.seq_length = 100
        self.n_chars = None
        self.n_vocab = None
        self.n_classes = None
        # This is the file to import training data from.
        self.filepath = 'inputs/input.txt'

    def create(self):
        # Define model.
        model = Sequential()

        model.add(LSTM(256,
                       input_shape=(self.input_shape)))
        model.add(Dropout(0.2))
        model.add(Dense(self.n_classes, activation='softmax'))

        # Compile model.
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model

    def load_data(self):
        raw_data = open(self.filepath).read()
        # Lowercase all input to make learning easier.
        raw_data = raw_data.lower()

        # Create mapping of unique chars to integers.
        chars = sorted(list(set(raw_data)))
        char_to_int = dict((c, i) for i, c in enumerate(chars))

        self.n_chars = len(raw_data)
        self.n_vocab = len(chars)

        X_train = []
        Y_train = []
        X_test = None
        Y_test = None

        # Grab training sequences from the corpus.
        for i in range(0, self.n_chars - self.seq_length, 1):
            seq_in = raw_data[i:i + self.seq_length]
            seq_out = raw_data[i + self.seq_length]
            X_train.append([char_to_int[char] for char in seq_in])
            Y_train.append(char_to_int[seq_out])

        # Convert lists to nparrays for returning.
        X_train = np.asarray(X_train)
        Y_train = np.asarray(Y_train)

        # Set the expected input shape to the model.
        self.input_shape = (self.seq_length, 1)

        return (X_train, Y_train), (X_test, Y_test)

    def process_inputs(self, X):
        # Reshape training data for input into the model.
        X_shape = (X.shape[0],) + self.input_shape
        X = np.reshape(X, X_shape)

        # Normalize inputs.
        X = X / float(self.n_vocab)

        return X

    def process_labels(self, Y):
        # Create one hot encodings.
        Y = np_utils.to_categorical(Y)
        self.n_classes = Y.shape[1]

        return Y
