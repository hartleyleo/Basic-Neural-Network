cmake_minimum_required(VERSION 3.10)
project(MyNeuralNet)

set(CMAKE_CXX_STANDARD 17)

find_package(pybind11 REQUIRED)

add_library(myneuralnet MODULE
    NeuralNetwork.cpp
    bindings.cpp
)

set_target_properties(myneuralnet PROPERTIES PREFIX "")

target_link_libraries(myneuralnet PRIVATE pybind11::module)
