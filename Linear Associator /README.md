
# Linear Associator Project

## Overview
This project includes the implementation and testing of the `LinearAssociator` class, which represents a linear associator neural network. The primary implementation is in `linear_associator.py`, and the tests are in `test_linear_associator.py`.

## Files in the Project
- `linear_associator.py`: Contains the `LinearAssociator` class with methods for the linear associator model.
- `test_linear_associator.py`: Includes tests for the `LinearAssociator` class using `pytest`.

## Usage
### Setting up the Linear Associator
To initialize the `LinearAssociator`, specify parameters like the number of input dimensions, number of nodes, and the transfer function. Example:

```python
from linear_associator import LinearAssociator
model = LinearAssociator(input_dimensions=2, number_of_nodes=4, transfer_function="Hard_limit")
```

### Running Tests
Run the tests using `pytest`:
```shell
pytest test_linear_associator.py
```

## Requirements
- Python 3.x
- NumPy
- pytest (for running tests)

## Installation
Install the required packages using pip:
```shell
pip install numpy pytest
```

