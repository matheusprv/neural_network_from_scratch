#ifndef NETWORK_HPP
#define NETWORK_HPP

#include "matrix_operations.hpp"
#include "activation.hpp"
#include "layer.hpp"
#include "loss.hpp"
#include <stack>

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

    void SGD(vector<matrix> X, vector<matrix> Y, int n_epochs=10, float learning_rate=0.001);
    
    float forwardPropagation(matrix x, matrix y, stack <matrix>* zs, stack<matrix>* as);

    void backpropagation(matrix y, stack <matrix>* zs, stack<matrix>* as, stack <matrix>* dW, stack<matrix>* dB);

    void updateWeightsAndBiases(const float& learning_rate, stack <matrix>* dW, stack<matrix>* dB);

    matrix make_prediction(matrix x);

    void evaluation(vector<matrix> X, vector<matrix> Y);

};


#endif