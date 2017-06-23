import argparse
from keras.datasets import mnist
from keras.models import load_model
from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import random

from model import Model


def process_inputs(X):
    # Reshape training data for input into the model.
    X_shape = (X.shape[0],) + Model.input_shape
    X = X.reshape(X_shape)

    # Normalize inputs by scaling from 0-255 to 0-1.
    X = X / 255

    return X


def process_labels(Y):
    # Create one hot encodings from class label vectors.
    Y = np_utils.to_categorical(Y)
    return Y


def main(args):
    # Load MNIST data.
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    # Process input data.
    X_train = process_inputs(X_train)
    X_test  = process_inputs(X_test)

    # Process input labels.
    Y_test  = process_labels(Y_test)
    Y_train = process_labels(Y_train)

    model = None
    train_model = args.train

    # Load preexisting model or create a new model.
    if (args.load_model is not None):
        model_path = Path(args.load_model)
        print('Loading model from {}.'.format(model_path))
        if (model_path.is_file()):
            model = load_model(model_path)
            print('Model loaded.'.format(model_path))
        else:
            print('Model not found.'.format(model_path))
    
    if model is None:
        print('No model loaded. Creating new model.')
        model = Model.create()
        train_model = True

    if train_model:
        print('Training model.')

        # Set default batch size to size of training set.
        if args.batch_size is None or args.batch_size <= 0:
            args.batch_size = len(X_train)

        model.fit(X_train,
                  Y_train,
                  validation_split=args.validation_split,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  verbose=2)

        # Evaluate model.
        scores = model.evaluate(X_test, Y_test, verbose=2)
        for i, label in enumerate(model.metrics_names):
            print('{}: {}'.format(label, scores[i]))
        print('Error rate: {:.2f}'.format(100 - (scores[1] * 100)))
        print('Training completed.')

        # Save model to file.
        if (args.save_model is not None):
            print('Saving model to file at {}.'.format(args.save_model))
            model.save(args.save_model)

    # Make predictions on test data.
    if (args.predict is not None):
        # Load testing data.
        print('Loading test data from {}.'.format(args.predict))
        test_filepath = args.predict
        test_data = np.genfromtxt(test_filepath, skip_header=1, delimiter=',')
        test_data = np.reshape(test_data, (len(test_data),) + Model.input_shape)
        print('Data loaded.')

        # Run the network against the testing data.
        print('Processing test data.')
        raw_preds = model.predict(test_data, verbose=1)
        print('Data processed.')

    # Save predictions to file.
    if (args.output is not None):
        print('Preparing prediction data.')
        predictions = {}
        for i, p in enumerate(raw_preds):
            predictions[i+1] = np.argmax(p)

        print('Saving predictions to file at {}.'.format(args.output))
        with open(args.output, 'w') as f:
            f.write('ImageId,Label\n')
            for k, p in predictions.items():
                f.write('{},{}\n'.format(k, p))
        f.close()


if __name__ == "__main__":
    default_model_file = 'model.h5'

    # Command line argument parsing.
    parser = argparse.ArgumentParser(description='Run MNIST classifier '
            'network.')
    parser.add_argument('--load-model', '-l',
            nargs='?',
            const=default_model_file,
            type=str,
            help='Load pre-existing Keras model from HDF5 file.',
            metavar='file_path')
    parser.add_argument('--train', '-t',
            action='store_true',
            help='Continue training model after loading.')
    parser.add_argument('--save-model', '-s',
            nargs='?',
            const=default_model_file,
            type=str,
            help='Save generated Keras model to file in HDF5 format.',
            metavar='file_path')
    parser.add_argument('--predict', '-p',
            nargs='?',
            const='test.csv',
            type=str,
            help='Retrieve input data to perform predictions on from file.',
            metavar='file_path')
    parser.add_argument('--output', '-o',
            nargs='?',
            const='submission.csv',
            type=str,
            help='Output final predictions to file in CSV format.',
            metavar='file_path')
    parser.add_argument('--validation-split', '-v',
            default=0.0,
            type=float,
            help='Ratio of training data to set aside for validation (range '
                    '0.0-1.0). Defaults to 0.0.',
            metavar='split_ratio')
    parser.add_argument('--epochs', '-e',
            default=1,
            type=int,
            help='Number of epochs to train for. Defaults to 1.',
            metavar='number_of_epochs')
    parser.add_argument('--batch-size', '-b',
            type=int,
            help='Size of batches to use during training. Defaults to size '
                'of entire training set.',
            metavar='batch_size')

    # Retrieve arguments passed in from command line.
    args = parser.parse_args()

    main(args)
