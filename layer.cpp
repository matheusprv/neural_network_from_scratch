#include "layer.hpp"

#include <random>
#include <chrono>

Layer :: Layer(int n_inputs, int n_neurons){
    this->n_inputs = n_inputs;
    this->n_neurons = n_neurons;

    this->weights = initializeWeights(n_inputs, n_neurons);
    this->bias = initializeBias(n_neurons);
}

Layer :: ~Layer(){}

/*
    Initialize the weights with normal distribution in range (-1.0, 1.0)
*/
vector<vector<float>> Layer :: initializeWeights(int inputs, int output){
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    default_random_engine generator(seed);
    normal_distribution<double> distribution(-1.0, 1.0);

    vector<vector<float>> weights(inputs, vector<float>(output, 0));

    for(int i=0; i<inputs; i++)
        for(int j=0; j<output; j++)
            weights[i][j] = distribution(generator);
        
    return weights;
}

/*
    Initialize the bias with zeros
*/
vector<float> Layer :: initializeBias(int n_neurons){
    vector<float> bias(n_neurons, 0);
    return bias;
}


int Layer :: getN_neurons() const{
    return this->n_neurons;
}

vector<vector<float>> Layer :: getWeights() const{
    return this->weights;
}
vector<float> Layer :: getBias() const{
    return this->bias;
}