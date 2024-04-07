
# CNN Project

## Overview
This project consists of a Convolutional Neural Network (CNN) implementation and its testing. The `CNN` class is defined in `cnn.py`, while `Gundala-04-03.py` likely demonstrates the usage or additional functionality, and `test_cnn.py` contains tests for the CNN implementation.

## Files in the Project
- `cnn.py`: Defines the `CNN` class with methods for building and manipulating a convolutional neural network.
- `Gundala-04-03.py`: Appears to be a script demonstrating the use or testing of the `CNN` class (further investigation required to understand its exact purpose).
- `test_cnn.py`: Contains unit tests for the `CNN` class using the `pytest` framework.

## Usage
### CNN Class
Instantiate and use the `CNN` class to create and work with convolutional neural networks. Example of setting up a CNN with an input layer and a convolutional layer:

```python
from cnn import CNN

# Create a CNN instance
cnn_model = CNN()

# Add input layer
cnn_model.add_input_layer(shape=(256, 256, 3), name="input")

# Add a convolutional layer
cnn_model.append_conv2d_layer(filters=32, kernel_size=(3, 3), activation='relu', name="conv1")
```

### Running the Example Script
Execute `Gundala-04-03.py` to see the `CNN` class in action (after understanding its purpose and requirements).

### Running Tests
To verify the implementation of the `CNN` class, run the tests in `test_cnn.py` using `pytest`:

```shell
pytest test_cnn.py
```

## Requirements
- Python 3.x
- NumPy
- TensorFlow
- Keras
- pytest (for running tests)

## Installation
Install the necessary dependencies using pip:

```shell
pip install numpy tensorflow keras pytest
```

