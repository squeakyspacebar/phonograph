# Phonograph

## Introduction

This is a simple command-line tool I hacked together to make it easier to train Keras models for the [Kaggle Digit Recognizer challenge](https://www.kaggle.com/c/digit-recognizer). Expanded to include some simple mucking around with LSTM networks. Written in Python 3.

Inspired by Jason Brownlee's great Keras tutorial articles:

- [Handwritten Digit Recognition](http://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/)
- [Text Generation](http://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/)

## Getting Started

It's recommended you use a virtual environment with this project, either through [virtualenv](https://virtualenv.pypa.io/en/latest/), [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/), or Python 3's included [venv](https://docs.python.org/3/library/venv.html) module.

The dependencies in `requirements.txt` don't explicitly include a [backend for Keras](https://keras.io/backend/), so you'll have to install Tensorflow or Theano manually and configure Keras to use it if necessary.

Use [Pip](https://pypi.python.org/pypi/pip) to install the Python package dependencies:

```
pip3 install -r requirements.txt
```

You can call the main script from the command line:

```
python3 phonograph.py
```

Command-line help is provided by:

```
python3 phonograph.py --help
```

### Using a pre-existing model

More about saving/loading Keras models [here](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model).

If a model file (in HDF5 format) of a pre-existing Keras model is provided, it is loaded to be used. Setting the `--load-model` or `-l` option without providing a path means that the script will look for `model.h5` in the current directory by default.

If the `--train` or `-t` flag is set, the model will go through a session of training. See the *Setting training parameters* section for more information.

From there, if the `--save-model` or `-s` option is provided, it will save the updated model to the path given, or `model.h5` in the current directory by default.

An example command to continue training from an existing model:

```
python3 phonograph.py --load-model shiny_model01.h5 --save-model shiny_model02.h5 --train --predict --output --validation-split 0.1 --epochs 10 --batch-size 32
```

### Training a new model

If no pre-existing model *file* is provided via the `--load-model` option, you'll have to specify which model factory you want to generate a new model from with the `--model` or `-m` option. Phonograph uses factories to provide the Keras model objects it uses. See [the section on adding new models](#adding-new-models) for further details.

There are currently two built-in options:

- `'cnn'` Convolutional
- `'lstm'` Long Short Term Memory

When selecting an untrained model, it will train by default whether or not the `--train` or `-t` option flag is set.

Saving the generated model to file works as mentioned using the `--save-model` or `-s` option.

An example command to train a new model:

```
python3 phonograph.py --model cnn --save-model --validation-split 0.1 --epochs 10 --batch-size 32
```

### Loading test data

If the `--predict` or `-p` option is set, the script will load the test data from the file at the path provided, or `data/test.csv` by default.

The script then runs `model.predict()` on the test data.

### Saving predictions

If the `--output` or `-o` option is set, the script will save the generated predictions to the file at the path provided, or `results/submission.csv` by default.

You can then submit your shiny new prediction set to Kaggle.

### Setting training parameters

You can set the training parameters by using these three options:

- `--validation-split <unit interval>` This sets the ratio of input data held for validation.
- `--epochs <integer>` This sets the number of epochs to train for.
- `--batch-size <integer>` This sets the training batch size.

## Modifying the Keras models

### Overview

The model factories are defined in the `factories` subdirectory and should inherit from the `ModelFactory` abstract base class in `model_factory.py` as an interface.

### Editing the models

Each model factory class should implement a method called `create()`. The actual Keras model being used should be defined within that method, and free to edit them as you wish.

### Adding new models

Create a model factory class called `Factory` that inherits from `ModelFactory` in a new file, then write the implementations for each required method. You may use whatever filenames you wish, but they are ultimately the labels of the models you can specify from the command-line; for example, to create and use a model called 'foobar', create a model factory in `foobar.py` and then specify the option `--model foobar` when running `phonograph.py`. Please see how the models created by the factories are used in `phonograph.py` and double-check against `model_factory.py` when creating a new model factory for more insight.

## Requirements

This application was written for Python 3. See `requirements.txt` for Python package requirements.

## Extra Notes

While it would be nice for this tool to be generalized to able to handle any data input and output format, that is beyond the scope of this project. If you need heavy-duty ETL, I personally suggest giving [Pentaho Kettle](http://community.pentaho.com/projects/data-integration/) a try. This project should be easy enough to adapt to specific cases, however.
