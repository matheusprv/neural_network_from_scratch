#ifndef LAYER_HPP
#define LAYER_HPP

#include <vector>
using namespace std;

class Layer{
    int n_inputs;
    int n_neurons;
    int n_bias;

    vector<vector<float>> weights;
    vector<float> bias;

    vector<vector<float>> initializeWeights(int inputs, int output);
    vector<float> initializeBias(int n_neurons);

public:
    Layer(int n_inputs=0, int n_neurons=0);
    virtual ~Layer();

    int getN_neurons() const;
    vector<vector<float>> getWeights() const;
    vector<float> getBias() const;

};

#endif