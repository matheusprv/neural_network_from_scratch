#ifndef LAYER_HPP
#define LAYER_HPP

#include <iostream>
#include <random>
#include <chrono>
#include <vector>

#include "utils.hpp"

class Layer{
    int inputs;
    int outputs;
    string activation;

    default_random_engine generator;

    matrix weights;
    matrix bias;

    matrix initializeWeights(int inputs, int outputs);
    matrix initializeBias(int outputs);

public:
    Layer(int inputs=0, int outputs=0, string activation="");
    virtual ~Layer();

    int getN_neurons() const;
    matrix getWeights() const;
    matrix getBias() const;
    
    void setWeights(matrix weights);
    void setBias(matrix bias);
    
    string getActivation() const;

    void printWeightsAndBias();

};

#endif