"""
This module handles the pipeline of training models, from instantiating
the models and calling the data loading methods, to supervising the
training, saving results to disk as necessary.
"""
import argparse
import importlib
from pathlib import Path
import sys
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np


def main(args):
    model = None
    train_model = args.train

    # Load the model factory.
    if args.model is not None:
        module_specifier = 'factories.{}'.format(args.model)
        factory_module = importlib.import_module(module_specifier)
        factory_class = getattr(factory_module, 'Factory')
        model_factory = factory_class()
        use_generators = False

        print('Preparing datasets.')
        try:
            # Load input data and data labels.
            (x_train, y_train), (x_test, y_test) = model_factory.load_data()

            # Process input data.
            x_train = model_factory.process_inputs(x_train)
            x_test = model_factory.process_inputs(x_test)

            # Process input labels.
            y_train = model_factory.process_labels(y_train)
            y_test = model_factory.process_labels(y_test)
            print('Datasets created.')
        except AttributeError:
            print('No data loading method found on factory. Trying generators.')
            train_generator, valid_generator = model_factory.get_generators()
            use_generators = True
            print('Generators created on datasets.')

    # Load preexisting model or create a new model.
    if args.load_model is not None:
        model_path = Path(args.load_model)
        print('Loading model from {}.'.format(model_path))

        if model_path.is_file():
            model = load_model(str(model_path))
            print('Model loaded.')
        else:
            print('Model not found.')

    if model is None:
        print('No model loaded. Creating new model.')
        model = model_factory.create()

        # Attempt to load model weights.
        if args.load_weights is not None:
            try:
                print('Loading model weights.')
                model.load_weights(args.load_weights)
                print('Weights loaded.')
            except ImportError:
                pass

        # Attempt to call finetuning if implemented.
        if args.finetune:
            try:
                print('Calling finetuning method.')
                model = model_factory.finetune(train_generator)
                print('Finetuning completed.')
            except AttributeError as e:
                raise Exception('Finetuning not implemented for model.') from e
        train_model = True

    if args.summary:
        model.summary()

    if train_model:
        callbacks_list = []

        # Define checkpoints.
        if args.disable_checkpoints is None:
            checkpoint_filepath = 'weights/weights-{epoch:02d}-{loss:.4f}.hdf5'
            checkpoint = ModelCheckpoint(
                checkpoint_filepath,
                monitor='loss',
                verbose=1,
                save_best_only=True,
                mode='min')
            callbacks_list = [checkpoint]

        print('Training model.')

        if use_generators:
            model.fit_generator(
                train_generator,
                steps_per_epoch=model_factory.train_steps,
                epochs=model_factory.epochs,
                validation_data=valid_generator,
                validation_steps=model_factory.valid_steps)
        else:
            # Set default batch size to size of training set.
            if args.batch_size is None or args.batch_size <= 0:
                args.batch_size = len(x_train)

            model.fit(
                x_train,
                y_train,
                epochs=args.epochs,
                batch_size=args.batch_size,
                callbacks=callbacks_list,
                verbose=2)

            # Evaluate model.
            scores = model.evaluate(x_test, y_test, verbose=2)
            for index, label in enumerate(model.metrics_names):
                print('{}: {}'.format(label, scores[index]))
            print('error: {:.2f}'.format(1 - scores[1]))

        print('Training completed.')

        # Save model to file.
        if args.save_model is not None:
            if args.save_model == 'default':
                args.save_model = '{}/model-{}-weights.hdf5'.format(
                    DEFAULT_MODEL_DIR,
                    args.model)
            print('Saving model to file at {}.'.format(args.save_model))
            model.save_weights(args.save_model)

    if args.evaluate:
        valid_generator = ImageDataGenerator()

        valid_generator.flow_from_directory(
            'data/validation',
            target_size=(224, 224),
            batch_size=args.batch_size,
            class_mode='categorical')

        valid_steps = 5000 // args.batch_size

        model.evaluate_generator(
            valid_generator,
            valid_steps)

    # Make predictions on test data.
    if args.predict is not None:
        try:
            if args.predict == 'default':
                args.predict = DEFAULT_INPUT_FILE

            # Load testing data.
            print('Attempting to load test data from {}.'.format(args.predict))
            test_data_path = Path(args.predict)

            if not test_data_path.exists():
                raise ValueError('No valid data found at {}.'.format(
                    test_data_path))

            if test_data_path.is_file():
                test_data = np.genfromtxt(test_data_path, skip_header=1, delimiter=',')
                test_data = np.reshape(
                    test_data,
                    (len(test_data),) + model.input_shape[1:])
                print('Data loaded.')

                # Run the network against the testing data.
                print('Processing test data.')
                raw_preds = model.predict(test_data, verbose=1)
                print('Data processed.')
            elif test_data_path.is_dir():
                test_datagen = ImageDataGenerator()

                test_generator = test_datagen.flow_from_directory(
                    test_path,
                    target_size=(224, 224),
                    batch_size=args.batch_size,
                    class_mode=None)

                test_steps = 12500 / args.batch_size

                print('Processing test data.')
                raw_preds = model.predict_generator(
                    test_generator,
                    test_steps)
                print('Data processed.')
        except:
            print('Failed. Unable to load test data.')
            raise

    # Save predictions to file.
    if args.output is not None:
        print('Preparing prediction data.')
        predictions = {}
        for index, prediction in enumerate(raw_preds):
            predictions[index + 1] = np.argmax(prediction)

        with open(args.output, 'w') as savefile:
            savefile.write('ImageId,Label\n')
            for key, prediction in predictions.items():
                savefile.write('{},{}\n'.format(key, prediction))
        savefile.close()


def create_parser():
    # Command line argument parsing.
    parser = argparse.ArgumentParser(
        description='Run Keras models.')
    parser.add_argument(
        '--load-model',
        '-l',
        nargs='?',
        const=DEFAULT_MODEL_FILE,
        type=str,
        help='Load pre-existing Keras model from HDF5 file.',
        metavar='file_path')
    parser.add_argument(
        '--model',
        '-m',
        nargs='?',
        type=str,
        help='Name of factory module to import.',
        metavar='module_name')
    parser.add_argument(
        '--load-weights',
        '-w',
        nargs='?',
        type=str,
        help='Load model weights from HDF5 file.',
        metavar='file_path')
    parser.add_argument(
        '--train',
        '-t',
        action='store_true',
        help='Run training if loading a model.')
    parser.add_argument(
        '--finetune',
        '-n',
        action='store_true',
        help='Run finetuning on a model.')
    parser.add_argument(
        '--save-model',
        '-s',
        nargs='?',
        const='default',
        type=str,
        help='Save generated Keras model to file in HDF5 format.',
        metavar='file_path')
    parser.add_argument(
        '--save-features',
        '-f',
        nargs='?',
        const=DEFAULT_OUTPUT_DIR,
        type=str,
        help='Save trained feature prediction arrays to file.',
        metavar='file_path')
    parser.add_argument(
        '--disable-checkpoints',
        '-d',
        action='store_true',
        help=(
            'Prevent saving generated weights to file in HDF5 format after '
            'each epoch.'))
    parser.add_argument(
        '--summary',
        '-y',
        action='store_true',
        help='Show summary of model before training.')
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Evaluate model.')
    parser.add_argument(
        '--predict',
        '-p',
        nargs='?',
        const='default',
        type=str,
        help='Retrieve input data to perform predictions on from file.',
        metavar='file_path')
    parser.add_argument(
        '--output',
        '-o',
        nargs='?',
        const=DEFAULT_OUTPUT_FILE,
        type=str,
        help='Output final predictions to file in CSV format.',
        metavar='file_path')
    parser.add_argument(
        '--validation-split',
        '-v',
        default=0.0,
        type=float,
        help='Ratio of training data to set aside for validation (range '
             '0.0-1.0). Defaults to 0.0.',
        metavar='split_ratio')
    parser.add_argument(
        '--epochs',
        '-e',
        default=1,
        type=int,
        help='Number of epochs to train for. Defaults to 1.',
        metavar='number_of_epochs')
    parser.add_argument(
        '--batch-size',
        '-b',
        nargs='?',
        const=32,
        type=int,
        help='Size of batches to use during training. Defaults to size '
             'of entire training set.',
        metavar='batch_size')

    return parser


def parse_args(parser):
    # Retrieve arguments passed in from command line.
    args = parser.parse_args()

    if args.load_model is None and args.model is None:
        print('Usage: --model must be given if --load_model is not used.')
        sys.exit(1)

    if args.load_model is not None and args.train:
        raise ValueError('Cannot train a model loaded from file.')

    return args


if __name__ == "__main__":
    DEFAULT_MODEL_DIR = 'models'
    DEFAULT_WEIGHT_DIR = 'weights'
    DEFAULT_INPUT_DIR = 'inputs'
    DEFAULT_OUTPUT_DIR = 'outputs'
    DEFAULT_MODEL_FILE = '{}/default.h5'.format(DEFAULT_MODEL_DIR)
    DEFAULT_INPUT_FILE = '{}/test.csv'.format(DEFAULT_INPUT_DIR)
    DEFAULT_OUTPUT_FILE = '{}/submission.csv'.format(DEFAULT_OUTPUT_DIR)

    main(parse_args(create_parser()))
