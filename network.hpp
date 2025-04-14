#ifndef NETWORK_HPP
#define NETWORK_HPP

#include "matrix_operations.hpp"
#include "activation.hpp"
#include "layer.hpp"
#include "loss.hpp"
#include <stack>

#define input vector<vector<float>> 
#define output vector<vector<float>> 

using namespace std;

class Network{
    int inputSize;
    int numLayers;
    vector<Layer> layers;    

    

public:
    Network(int inputSize);
    virtual ~Network();

    void addLayer(int n_neurons, string activation="");

    void printNetwork();


    void SGD(vector<input> X, vector<output> Y, int n_epochs=10, float learning_rate=0.001);

};


#endif