from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils import np_utils

from model_factory import ModelFactory


class Factory(ModelFactory):
    def __init__(self):
        self.img_width = 224
        self.img_height = 224
        self.img_channels = 3
        # Set the expected shape of input here.
        # TensorFlow ordering. Reverse for use with Theano.
        self.input_shape = (self.img_channels, self.img_width, self.img_height)
        self.batch_size = 1
        self.epochs = 1
        self.training_samples = 20000
        self.validation_samples = 5000
        self.training_data_dir = 'data/train'
        self.validation_data_dir = 'data/validation'

    def create(self):
        # Define model.
        model = Sequential()

        model.add(Conv2D(32, (3, 3), activation='relu',
                         input_shape=self.input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))

        # Compile model.
        model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

        return model

    """
    This is the augmentation configuration we will use for training.
    """
    def get_generators(self):
        training_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

        testing_datagen = ImageDataGenerator(rescale=1./255)

        training_generator = self.get_batches(self.training_data_dir)

        validation_generator = self.get_batches(self.validation_data_dir)

        return training_generator, validation_generator

    def get_batches(self, path, gen=ImageDataGenerator(), shuffle=True):
        return gen.flow_from_directory(
            path,
            target_size=(self.img_width, self.img_height),
            batch_size=self.batch_size,
            shuffle=shuffle,
            class_mode='binary')
