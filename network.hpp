#ifndef NETOWROK_HPP
#define NETOWROK_HPP

#include <vector>
#include <stdexcept>
#include "layer.hpp"
#include <iostream>

using namespace std;

class Network{
    int inputSize;
    int numLayers;
    vector<Layer> layers;    

public:
    Network(int inputSize);
    virtual ~Network();

    vector<vector<float>> matrixMultiplication(vector<vector<float>> a, vector<vector<float>> b);

    vector<vector<float>> addBias(vector<vector<float>> a, vector<float> b);

    vector<vector<float>> ReLU(vector<vector<float>> z);

    vector<vector<float>> feedLayer(vector<vector<float>> X, Layer layer);

    vector<vector<float>> feedForward(vector<vector<float>> X);

    void addLayer(int n_neurons, string activation="");

    float mse(vector<float> yTrue, vector<float> yHat);

    void printNetwork();

};


#endif