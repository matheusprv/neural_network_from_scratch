#include "network.hpp"

Network :: Network(int inputSize){
    this->inputSize = inputSize;
    this->numLayers = 0;
}

Network :: ~Network(){}

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

void Network :: printNetwork(){
    for(Layer temp : this->layers){
        temp.printWeightsAndBias();
        cout << " " << endl;
    }
}


void Network :: SGD(vector<input> X, vector<output> Y, int n_epochs, float learning_rate){
    if(X.size() != Y.size()) throw invalid_argument("X and Y have different sizes");

    int L = this->layers.size();
    
    for(int epoch=1; epoch <= n_epochs; epoch++){
        float loss = 0.0;
        cout << "Starting epoch " << epoch << endl;

        for(size_t i = 0; i < X.size(); i++){
            input x = X[i];
            output y = Y[i];

            
            stack <vector<vector<float>>> dW;
            stack <vector<vector<float>>> dB;

            stack <vector<vector<float>>> zs;
            stack <vector<vector<float>>> as;

            as.push(x);
            // forward
            vector<vector<float>> z;
            vector<vector<float>> a = x;
            for(Layer layer : this->layers){
                z = matrixMultiplication(layer.getWeights(), a);
                z = matrixAddition(z, layer.getBias());

                if(layer.getActivation()=="relu")
                    a = ReLU(z);
                else
                    a = z;
                
                zs.push(z);
                as.push(a);
            }

            loss += squaredError(a, y);

            // backprop
            vector<vector<float>> delta = hadamardProduct(
                seDerivative(as.top(), y),
                linearDerivative(zs.top())
            );
            as.pop(); zs.pop();

            dB.push(delta);
            dW.push(matrixMultiplication(delta, transposeMatrix(as.top())));
            as.pop();
            
            for(int l = L-2; l >= 0; l--){
                vector<vector<float>> activationDerivative;
                if(this->layers[l].getActivation() == "relu")
                    activationDerivative = ReLUDerivative(zs.top());
                else
                    activationDerivative = linearDerivative(zs.top());

                delta = hadamardProduct(
                    matrixMultiplication(
                        transposeMatrix(this->layers[l+1].getWeights()),
                        delta
                    ),
                    activationDerivative
                );

                dB.push(delta);
                dW.push(matrixMultiplication(delta, transposeMatrix(as.top())));

                as.pop(); zs.pop();
            }

            // update weights and biases
            for(Layer &layer : this->layers){

                layer.setWeights(
                    matrixAddition(
                        layer.getWeights(),
                        multiplicationByScalar(-learning_rate, dW.top())
                    )
                );

                layer.setBias(
                    matrixAddition(
                        layer.getBias(),
                        multiplicationByScalar(-learning_rate, dB.top())
                    )
                );

                dB.pop();
                dW.pop();
            }
        }
    
        loss /= X.size();
        cout << "mse: " << loss << endl;
    }
}
