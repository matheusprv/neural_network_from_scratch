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

void Network :: SGD(vector<matrix> X, vector<matrix> Y, int n_epochs, float learning_rate){
    if(X.size() != Y.size()) throw invalid_argument("X and Y have different sizes");
    
    for(int epoch=1; epoch <= n_epochs; epoch++){
        float loss = 0.0;
        cout << "Starting epoch " << epoch << endl;

        for(size_t i = 0; i < X.size(); i++){
            matrix x = X[i];
            matrix y = Y[i];
            
            stack <matrix> zs;
            stack <matrix> as;
            
            // forward
            loss += this->forwardPropagation(x, y, &zs, &as);
            
            // backprop
            stack <matrix> dW;
            stack <matrix> dB;
            this->backpropagation(y, &zs, &as, &dW, &dB);

            // update weights and biases
            this->updateWeightsAndBiases(learning_rate, &dW, &dB);
        }
    
        loss /= X.size();
        cout << "mse: " << loss << endl;
    }
}

float Network :: forwardPropagation(matrix x, matrix y, stack <matrix>* zs, stack<matrix>* as){
    as->push(x);
    matrix z;
    matrix a = x;
    for(Layer layer : this->layers){
        z = matrixMultiplication(layer.getWeights(), a);
        z = matrixAddition(z, layer.getBias());

        if(layer.getActivation()=="relu")
            a = ReLU(z);
        else
            a = z;
        
        zs->push(z);
        as->push(a);
    }

    return squaredError(a, y);
}

void Network :: backpropagation(matrix y, stack <matrix>* zs, stack<matrix>* as, stack <matrix>* dW, stack<matrix>* dB){
   
    int L = this->layers.size();

    matrix delta = hadamardProduct(
        seDerivative(as->top(), y),
        linearDerivative(zs->top())
    );
    as->pop(); zs->pop();

    dB->push(delta);
    dW->push(matrixMultiplication(delta, transposeMatrix(as->top())));
    as->pop();
    
    for(int l = L-2; l >= 0; l--){
        matrix activationDerivative;
        if(this->layers[l].getActivation() == "relu")
            activationDerivative = ReLUDerivative(zs->top());
        else
            activationDerivative = linearDerivative(zs->top());

        delta = hadamardProduct(
            matrixMultiplication(
                transposeMatrix(this->layers[l+1].getWeights()),
                delta
            ),
            activationDerivative
        );

        dB->push(delta);
        dW->push(matrixMultiplication(delta, transposeMatrix(as->top())));

        as->pop(); zs->pop();
    }

}

void Network :: updateWeightsAndBiases(const float& learning_rate, stack <matrix>* dW, stack<matrix>* dB){
    for(Layer &layer : this->layers){

        layer.setWeights(
            matrixAddition(
                layer.getWeights(),
                multiplicationByScalar(-learning_rate, dW->top())
            )
        );

        layer.setBias(
            matrixAddition(
                layer.getBias(),
                multiplicationByScalar(-learning_rate, dB->top())
            )
        );

        dB->pop();
        dW->pop();
    }
}

matrix Network :: make_prediction(matrix x){
    matrix z;
    matrix a = x;
    for(Layer layer : this->layers){
        z = matrixMultiplication(layer.getWeights(), a);
        z = matrixAddition(z, layer.getBias());

        if(layer.getActivation()=="relu")
            a = ReLU(z);
        else
            a = z;
    }

    return a;
}

void Network :: evaluate(vector<matrix> X, vector<matrix> Y){

    int N = static_cast<int>(X.size());    
    float loss = 0.0f;
    
    for(size_t i = 0u; i < X.size(); i++){
        matrix x = X[i];
        matrix y = Y[i];
        
        matrix yPred = this->make_prediction(x);

        loss += squaredError(yPred, y);
    }

    loss /= N;
    cout << "Test loss: " << loss << endl;
}