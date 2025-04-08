#include "network.hpp"

Network :: Network(int inputSize){
    this->inputSize = inputSize;
    this->numLayers = 0;
}

Network :: ~Network(){}

vector<vector<float>> Network :: matrixMultiplication(vector<vector<float>> a, vector<vector<float>> b){
    int rowsA = a.size();
    int columnsA = a[0].size();

    int rowsB = b.size();
    int columnsB = b[0].size();

    if(columnsA != rowsB){
        throw invalid_argument(
            "Matrix A has shape (" + to_string(rowsA) + "," + to_string(columnsA) +") and " +
            "matrix B has shape (" + to_string(rowsB) + "," + to_string(columnsB) +").\n" +
            "The number of columns of matrix A must match the number of rows of matrix B"
        );
    }

    vector<vector<float>> output(rowsA, vector<float>(columnsB, 0));

    for(int i=0; i < rowsA; i++)
        for(int j=0; j < columnsB; j++)
            for(int k=0; k < columnsA; k++) // columnsA = rowsB 
                output[i][j] += a[i][k] * b[k][j];
            
    return output;   
}

vector<vector<float>> Network :: addBias(vector<vector<float>> a, vector<float> b){
    int rowsA = a.size();
    int columnsA = a[0].size();

    int columnsB = b.size();

    if(rowsA != 1 || columnsA != columnsB){
        throw invalid_argument(
            "Matrix A has shape (" + to_string(rowsA) + "," + to_string(columnsA) +") and " +
            "matrix B has shape (" + to_string(1) + "," + to_string(columnsB) +").\n" +
            "The matrices must have the same shape"
        );
    }

    vector<vector<float>> output(rowsA, vector<float>(columnsB, 0));

    for(int i=0; i<rowsA; i++)
        for(int j=0; j<columnsA; j++)
            output[i][j] = a[i][j] + b[j];
    
    return output;
}

vector<vector<float>> Network :: ReLU(vector<vector<float>> z){
    int rows = z.size();
    int columns = z[0].size();

    for(int i=0; i<rows; i++)
        for(int j=0; j<columns; j++)
            if(z[i][j] < 0)
                z[i][j] = 0;

    return z;
}

vector<vector<float>> Network :: feedLayer(vector<vector<float>> X, Layer layer){    
    vector<vector<float>> W = layer.getWeights();
    vector<float> b = layer.getBias();
    
    vector<vector<float>> z = matrixMultiplication(X, W);
    z = addBias(z, b);

    if(layer.getActivation() == "relu"){
        vector<vector<float>> a = ReLU(z);
        return a;
    }
    else{
        return z;
    }
    
}

vector<vector<float>> Network :: feedForward(vector<vector<float>> X){
    vector<vector<float>> temp = this->feedLayer(X, this->layers[0]);
    
    for(int i=1; i<this->numLayers; i++){
        temp = feedLayer(temp, this->layers[i]);
    }

    return temp;    
}

void Network :: addLayer(int n_neurons, string activation){
    int neuronInputs = 0;

    if(this->numLayers == 0)
        // Size of the input vector
        neuronInputs = this->inputSize;
    else
        // Number of neurons from the previous layer
        neuronInputs = this->layers[this->numLayers - 1].getN_neurons();
    
    Layer newLayer(neuronInputs, n_neurons, activation);
    this->layers.push_back(newLayer);
    this->numLayers++;
}

float Network :: mse(vector<float> yTrue, vector<float> yPred){
    int n_yTrue = yTrue.size();
    int n_yPred = yPred.size();

    if (n_yTrue != n_yPred) throw invalid_argument("yTrue ("+ to_string(n_yTrue) +") and yPred "+ to_string(n_yPred) +" have different number of itens");
    
    float summation = 0;
    float temp = 0.0;
    for(int i = 0; i < n_yTrue; i++){
        temp = yTrue[i] - yPred[i];
        summation += temp*temp;
    }

    summation = summation / n_yTrue;
    return summation;
}


void Network :: printNetwork(){
    for(Layer temp : this->layers){
        temp.printWeightsAndBias();
        cout << " " << endl;
    }
}