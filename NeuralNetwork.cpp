#include "NeuralNetwork.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <ctime>

double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

double sigmoidDerivative(double x) {
    return x * (1.0 - x);
}

// ----------------- Neuron -----------------
Neuron::Neuron(int numInputs) {
    weights.resize(numInputs);
    for (double& w : weights) {
        w = ((double) std::rand() / RAND_MAX) - 0.5;
    }
    bias = ((double) std::rand() / RAND_MAX) - 0.5;
} 

double Neuron::activate(const std::vector<double>& inputs) {
    double sum = bias;
    for (std::size_t i = 0; i < inputs.size(); ++i)
        sum += weights[i] * inputs[i];
    output = sigmoid(sum);
    return output;
}

// ----------------- Layer -----------------
Layer::Layer(int numNeurons, int numInputsPerNeuron) {
    neurons.reserve(numNeurons);
    for (int i = 0; i < numNeurons; ++i) {
        neurons.emplace_back(numInputsPerNeuron);
    }
}

std::vector<double> Layer::forward(const std::vector<double>& inputs) {
    std::vector<double> outputs;
    outputs.reserve(neurons.size());
    for (auto& n : neurons)
        outputs.push_back(n.activate(inputs));
    return outputs;
}

// ----------------- NeuralNetwork -----------------
NeuralNetwork::NeuralNetwork(const std::vector<int>& layerSizes) {
    if (layerSizes.size() < 2)
        throw std::invalid_argument("Network must have at least 2 layers.");
    for (std::size_t i = 1; i < layerSizes.size(); ++i)
        layers.emplace_back(Layer(layerSizes[i], layerSizes[i - 1]));
}

std::vector<double> NeuralNetwork::forward(const std::vector<double>& inputs) {
    std::vector<double> activations = inputs;
    for (auto& layer : layers)
        activations = layer.forward(activations);
    return activations;
}

void NeuralNetwork::train(const std::vector<std::vector<double>>& inputs,
                          const std::vector<std::vector<double>>& targets,
                          int epochs) {

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double totalError = 0.0;

        for (std::size_t i = 0; i < inputs.size(); ++i) {
            // Forward pass
            std::vector<std::vector<double>> layerOutputs;
            layerOutputs.reserve(layers.size() + 1);
            layerOutputs.push_back(inputs[i]);

            std::vector<double> activations = inputs[i];
            for (auto& layer : layers) {
                activations = layer.forward(activations);
                layerOutputs.push_back(activations);
            }

            // Error
            const std::vector<double>& outputs = layerOutputs.back();
            std::vector<double> outputErrors(outputs.size());
            for (std::size_t j = 0; j < outputs.size(); ++j) {
                double error = targets[i][j] - outputs[j];
                outputErrors[j] = error;
                totalError += error * error;
            }

            // Backprop
            Layer& lastLayer = layers.back();
            for (std::size_t j = 0; j < lastLayer.neurons.size(); ++j)
                lastLayer.neurons[j].delta = outputErrors[j] * sigmoidDerivative(lastLayer.neurons[j].output);

            for (int l = static_cast<int>(layers.size()) - 2; l >= 0; --l) {
                Layer& current = layers[l];
                Layer& next = layers[l + 1];
                for (std::size_t j = 0; j < current.neurons.size(); ++j) {
                    double error = 0.0;
                    for (std::size_t k = 0; k < next.neurons.size(); ++k)
                        error += next.neurons[k].weights[j] * next.neurons[k].delta;
                    current.neurons[j].delta = error * sigmoidDerivative(current.neurons[j].output);
                }
            }

            // Update weights
            for (std::size_t l = 0; l < layers.size(); ++l) {
                const std::vector<double>& inputsToLayer = layerOutputs[l];
                for (auto& neuron : layers[l].neurons) {
                    for (std::size_t w = 0; w < neuron.weights.size(); ++w)
                        neuron.weights[w] += learningRate * neuron.delta * inputsToLayer[w];
                    neuron.bias += learningRate * neuron.delta;
                }
            }
        }

        std::cout << "Epoch " << epoch + 1
                  << ", Error: " << totalError / inputs.size() << std::endl;
    }
}

// ----------------- Save / Load -----------------
bool NeuralNetwork::save(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) return false;

    file << layers.size() << "\n";
    for (const auto& layer : layers)
        file << layer.neurons.size() << " ";
    file << "\n";

    for (const auto& layer : layers) {
        for (const auto& neuron : layer.neurons) {
            for (double w : neuron.weights)
                file << w << " ";
            file << neuron.bias << "\n";
        }
    }

    return true;
}

bool NeuralNetwork::load(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) return false;

    std::size_t numLayers;
    file >> numLayers;

    std::vector<int> layerSizes(numLayers);
    for (std::size_t i = 0; i < numLayers; ++i)
        file >> layerSizes[i];

    layers.clear();
    for (std::size_t i = 1; i < layerSizes.size(); ++i)
        layers.emplace_back(Layer(layerSizes[i], layerSizes[i - 1]));

    for (auto& layer : layers) {
        for (auto& neuron : layer.neurons) {
            for (double& w : neuron.weights)
                file >> w;
            file >> neuron.bias;
        }
    }

    return true;
}
