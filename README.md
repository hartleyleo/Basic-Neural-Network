# Basic Neural Network C++ Library with Python Bindings

This project was authored by [Leon Hartley](https://github.com/hartleyleo).

This project implements a **simple, trainable neural network in C++** with **Python bindings via pybind11**. The network supports a customizable number of inputs and outputs, trains using backpropagation with a sigmoid activation, and can save/load its weights to/from a file. Since it has python bindings, it can be used directly from Python as a library.

---

## Features

* Fully connected feedforward neural network
* Configurable layer sizes
* Sigmoid activation function
* Trainable via backpropagation
* Save and load network weights
* Exposed to Python via pybind11 for easy integration

---

## Folder Structure

```
Basic-Neural-Network/
├── NeuralNetwork.h          # Network, layer, neuron definitions
├── NeuralNetwork.cpp        # Network implementation
├── bindings.cpp             # Python bindings via pybind11
├── CMakeLists.txt           # CMake build
```

---

## Prerequisites

* **C++ compiler** (supporting C++17)
* **CMake** ≥ 3.10
* **Python 3** (preferably in a virtual environment)
* **pip packages**:

  * `pybind11`

---

## For using the library in Python

1. Make sure the compiled `.so` file is in the **same directory** as your Python script (or add its path to `PYTHONPATH`).

2. Example usage:

```python
import myneuralnet
import numpy as np
from keras.datasets import mnist  # Optional for dataset loading

# Create network
nn = myneuralnet.NeuralNetwork([784, 128, 10])
n.learning_rate = 0.1

# Load MNIST dataset (28x28 grayscale images)
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape((-1, 28*28)) / 255.0
x_test  = x_test.reshape((-1, 28*28)) / 255.0

# Convert labels to one-hot
targets = np.zeros((y_train.shape[0], 10))
for i, lbl in enumerate(y_train):
    targets[i, lbl] = 1.0

inputs = x_train.tolist()
trg = targets.tolist()

# Train network
nn.train(inputs, trg, epochs=100)

# Save network to a file
success = nn.save("mnist_network.txt")
print("Saved successfully:", success)

# Load network into a new object
nn2 = myneuralnet.NeuralNetwork([784, 128, 10])
n2.load("mnist_network.txt")
print("Loaded successfully!")

# Make predictions
prediction = nn2.forward(inputs[0])
print("First image prediction:", prediction)
```
