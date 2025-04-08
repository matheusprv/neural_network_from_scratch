#include "layer.hpp"


Layer :: Layer(int n_inputs, int n_neurons, string activation){
    this->n_inputs = n_inputs;
    this->n_neurons = n_neurons;
    this->activation = activation;

    this->weights = initializeWeights(n_inputs, n_neurons);
    this->bias = initializeBias(n_neurons);
}

Layer :: ~Layer(){}

/*
    Initialize the weights with normal distribution in range (-1.0, 1.0)
*/
vector<vector<float>> Layer :: initializeWeights(int inputs, int n_neurons){
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    default_random_engine generator(seed);
    normal_distribution<double> distribution(0, 0.5);

    vector<vector<float>> weights(inputs, vector<float>(n_neurons, 0));

    for(int i=0; i<inputs; i++)
        for(int j=0; j<n_neurons; j++)
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


void Layer :: printWeightsAndBias(){
    cout << "Weights: " << endl;
    for(int i=0; i < this->n_inputs; i++){
        for(int j = 0; j< this->n_neurons; j++){
            cout << this->weights[i][j] << " ";
        }
        cout << endl;
    }
    cout << "\nBias: " << endl;
    for(int j = 0; j< this->n_neurons; j++){
        cout << this->bias[j] << " ";
    }
    cout << "\n";
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
string Layer :: getActivation() const{
    return this->activation;
}