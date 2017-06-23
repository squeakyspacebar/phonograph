# Phonograph

## Introduction

This is a simple command-line tool I hacked together to make it easier to train
Keras models for the [Kaggle Digit Recognizer challenge](https://www.kaggle.com/c/digit-recognizer).
Currently tied to processing the MNIST database of handwritten digits fairly
specifically. Written in Python 3.

## Getting Started

If you're interested in using this code, it probably doesn't need to be said,
but use [Pip](https://pypi.python.org/pypi/pip) to install the Python package
dependencies:

```
pip3 install requirements.txt
```

You can call the main script from the command line:

```
python3 phonograph.py
```

Command-line help is provided by:

```
python3 phonograph.py --help
```

An example — in long format — of a standard command:

```
python3 phonography.py --load-model shiny_model01.h5 --save-model  
shiny_model02.h5 --train --predict --output --validation-split 0.1 --epochs 10  
--batch-size 32
```

### Using a pre-existing model

More about saving/loading Keras models [here](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model).

If a model file (in HDF5 format) of a pre-existing Keras model is provided,
it is loaded to be used. Setting the `--load-model` or `-l` option without
providing a path means that the script will look for `model.h5` in the current
directory by default.

If the `--train` or `-t` flag is set, the model will go through a session of
training. See the *Setting training parameters* section for more information.

From there, if the `--save-model` or `-s` option is provided, it will save the
updated model to the path given, or `model.h5` in the current directory by
default.

### Creating a new model

If no model file of a pre-existing Keras model is provided, `phonograph.py`
uses whatever model `Model.create()` provides. You can define the model you want
to use in `model.py` (just make sure `Model.create()` is the method that returns
the model).

After creating a model, it will automatically train, whether or not the option
flag is set.

Saving the generated model works as mentioned in the above section.

### Loading test data

If the `--predict` or `-p` option is set, the script will load the test data
from the file at the path provided, or `test.csv` in the current directory by
default.

The script then runs `model.predict()` on the test data.

### Saving predictions

If the `--output` or `-o` option is set, the script will save the generated
predictions to the file at the path provided, or `submission.csv` in the current
directory by default.

You can then submit your shiny new prediction set to Kaggle.

### Setting training parameters

You can set the training parameters by using these three options:

- `--validation-split <unit interval>` This sets the ratio of input data held
for validation.
- `--epochs <integer>` This sets the number of epochs to train for.
- `--batch-size <integer>` This sets the training batch size.

## Extra Notes

While it would be nice for this tool to be generalized to able to handle any
data input and output format, that is beyond the scope of this project. If you
need heavy-duty ETL, I personally suggest giving [Pentaho Kettle](http://community.pentaho.com/projects/data-integration/)
a try. This project should be easy enough to adapt to specific cases, however.
