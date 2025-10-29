#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "NeuralNetwork.h"

namespace py = pybind11;

PYBIND11_MODULE (myneuralnet, m) {
    m.doc() = "Simple trainable neural network implemented in C++";

    py::class_<NeuralNetwork>(m, "NeuralNetwork")
        .def(py::init<const std::vector<int>&>())
        .def("train", &NeuralNetwork::train)
        .def("forward", &NeuralNetwork::forward)
        .def("save", &NeuralNetwork::save)
        .def("load", &NeuralNetwork::load)
        .def_readwrite("learning_rate", &NeuralNetwork::learningRate);
}