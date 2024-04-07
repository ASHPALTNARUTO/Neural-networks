
# Single Layer Neural Network

## Description
This project contains the implementation of a single-layer neural network (`SingleLayerNN`) in Python using NumPy. The `single_layer_nn.py` file defines the neural network's structure and basic functionalities such as weight initialization, feedforward computation, and training mechanisms.

The `test_single_layer_nn.py` file includes tests for the neural network, ensuring that the implementation is correct and the network behaves as expected.

## Installation

To run this project, you need Python and some specific packages. Follow these steps to set up your environment:

1. Ensure you have Python installed on your machine. Python 3.8 or higher is recommended.
2. Install the required Python packages using pip. Navigate to the project directory and run:

   ```bash
   pip install numpy pytest
   ```

This will install NumPy for numerical computations in the neural network and pytest for running the test suite.

## Running the Tests

To verify the correctness of the neural network implementation, execute the test suite using pytest. In the terminal, navigate to the project directory and run the following command:

```bash
pytest test_single_layer_nn.py
```

This command will execute all test cases defined in `test_single_layer_nn.py`, testing various aspects of the `SingleLayerNN` class.

## Usage

The main functionality is encapsulated in the `SingleLayerNN` class. While there is no main execution script for the neural network, you can utilize the class in your own Python scripts by importing it:

```python
from single_layer_nn import SingleLayerNN
```

After importing, you can instantiate and use the `SingleLayerNN` class as per your requirements for neural network experiments or further development.

