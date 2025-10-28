#pragma once
#include <vector>
#include <string>

struct Neuron {
    std::vector<double> weights;
    double bias = 0.0;
    double output = 0.0;
    double delta = 0.0;
    explicit Neuron(int numInputs);
    double activate(const std::vector<double>& inputs);
};

struct Layer {
    std::vector<Neuron> neurons;
    Layer(int numNeurons, int numInputsPerNeuron);
    std::vector<double> forward(const std::vector<double>& inputs);
};

struct NeuralNetwork {
    std::vector<Layer> layers;
    double learningRate = 0.1;

    explicit NeuralNetwork(const std::vector<int>& layerSizes);

    std::vector<double> forward(const std::vector<double>& inputs);
    void train(const std::vector<std::vector<double>>& inputs,
               const std::vector<std::vector<double>>& targets,
               int epochs);

    // Save/Load functions
    bool save(const std::string& filename) const;
    bool load(const std::string& filename);
};
