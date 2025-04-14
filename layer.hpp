#ifndef LAYER_HPP
#define LAYER_HPP

#include <iostream>
#include <random>
#include <chrono>
#include <vector>
using namespace std;

class Layer{
    int inputs;
    int outputs;
    string activation;

    default_random_engine generator;

    vector<vector<float>> weights;
    vector<vector<float>> bias;

    vector<vector<float>> initializeWeights(int inputs, int outputs);
    vector<vector<float>> initializeBias(int outputs);

public:
    Layer(int inputs=0, int outputs=0, string activation="");
    virtual ~Layer();

    int getN_neurons() const;
    vector<vector<float>> getWeights() const;
    vector<vector<float>> getBias() const;
    
    void setWeights(vector<vector<float>> weights);
    void setBias(vector<vector<float>> bias);
    
    string getActivation() const;

    void printWeightsAndBias();

};

#endif