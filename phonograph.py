import argparse
from keras.models import load_model
import numpy as np
from pathlib import Path

from model_factory import ModelFactory


def main(args):
    model_factory = ModelFactory()

    # Load input data and data labels.
    (X_train, Y_train), (X_test, Y_test) = model_factory.load_data()

    # Process input data.
    if X_train is not None:
        X_train = model_factory.process_inputs(X_train)
    if X_test is not None:
        X_test  = model_factory.process_inputs(X_test)

    # Process input labels.
    if Y_train is not None:
        Y_train = model_factory.process_labels(Y_train)
    if Y_test is not None:
        Y_test  = model_factory.process_labels(Y_test)

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
        model = model_factory.create()
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
        test_data = np.reshape(test_data, (len(test_data),) + model.input_shape)
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
    default_model_dir   = 'models'
    default_weight_dir  = 'weights'
    default_input_dir   = 'inputs'
    default_output_dir  = 'outputs'
    default_model_file  = '{}/default.h5'.format(default_model_dir)
    default_input_file  = '{}/test.csv'.format(default_input_dir)
    default_output_file = '{}/submission.csv'.format(default_output_dir)

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
            const=default_input_file,
            type=str,
            help='Retrieve input data to perform predictions on from file.',
            metavar='file_path')
    parser.add_argument('--output', '-o',
            nargs='?',
            const=default_output_file,
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
