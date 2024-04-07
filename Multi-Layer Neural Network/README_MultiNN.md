
# MultiNN Project

## Overview
This project includes the implementation and testing of the `MultiNN` class, which represents a multi-layer neural network. The `MultiNN` class is implemented in `multinn.py`, and the tests for this class are in `test_multinn.py`.

## Files in the Project
- `multinn.py`: Contains the `MultiNN` class with methods to initialize the network and add layers with specified configurations.
- `test_multinn.py`: Includes tests for the `MultiNN` class using the `pytest` framework, focusing on the correct configuration and dimensions of the network layers.

## Usage
### MultiNN Class
The `MultiNN` class allows for creating a customizable multi-layer neural network. Initialize the network with the input dimension, and then add layers specifying the number of nodes and the transfer function as needed.

```python
from multinn import MultiNN

# Initialize the neural network with the input dimension
network = MultiNN(input_dimension=4)

# Add layers to the network
network.add_layer(num_nodes=10, transfer_function='ReLU')
network.add_layer(num_nodes=20, transfer_function='Sigmoid')
```

### Running Tests
Ensure the functionality of the `MultiNN` class by running the tests using `pytest`:

```shell
pytest test_multinn.py
```

## Requirements
- Python 3.x
- NumPy
- TensorFlow
- pytest (for running tests)

## Installation
Install the required packages using pip:

```shell
pip install numpy tensorflow pytest
```

