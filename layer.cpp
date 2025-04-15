#include "layer.hpp"


Layer :: Layer(int inputs, int outputs, string activation){
    this->inputs = inputs;
    this->outputs = outputs;
    this->activation = activation;

    this->weights = initializeWeights(inputs, outputs);
    this->bias = initializeBias(outputs);

    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    //seed = 42;
    this->generator = default_random_engine (seed);

}

Layer :: ~Layer(){}

/*
    Initialize the weights with normal distribution in range (-1.0, 1.0)
*/
vector<vector<float>> Layer :: initializeWeights(int inputs, int outputs){

    normal_distribution<double> distribution(0, 0.5);

    vector<vector<float>> weights(outputs, vector<float>(inputs, 0));

    for(int i=0; i<outputs; i++)
        for(int j=0; j<inputs; j++)
            weights[i][j] = distribution(this->generator);
        
    return weights;
}

/*
    Initialize the bias with zeros
*/
vector<vector<float>> Layer :: initializeBias(int outputs){
    vector<vector<float>> bias(outputs, vector<float>(1, 0));
    return bias;
}


int Layer :: getN_neurons() const{
    return this->outputs;
}

vector<vector<float>> Layer :: getWeights() const{
    return this->weights;
}
vector<vector<float>> Layer :: getBias() const{
    return this->bias;
}

void Layer :: setWeights(vector<vector<float>> weights){
    this->weights = weights;
}
void Layer ::  setBias(vector<vector<float>> bias){
    this->bias = bias;
}


string Layer :: getActivation() const{
    return this->activation;
}

void Layer :: printWeightsAndBias(){
    cout << "Weights: " << endl;
    for(int i = 0; i < outputs; i++) {
        for(int j = 0; j < inputs; j++) {
            cout << weights[i][j] << " ";
        }
        cout << endl;
    }    
    cout << "\nBias: " << endl;
    for(int j = 0; j< this->outputs; j++){
        cout << this->bias[j][0] << " ";
    }
    cout << "\n";
}

