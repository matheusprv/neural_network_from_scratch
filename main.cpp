#include <iostream>
#include <vector>
#include <stdexcept>

using namespace std;

#include "network.hpp"


int main() {

    vector<vector<float>> X = {
        {0.96639924, 0.82608733, 0.68022676, 0.83469193, 0.86759439}
    };

    vector<float> Y = {1.0};

    Network network(5);
    network.addLayer(3, "relu");
    network.addLayer(3, "relu");
    network.addLayer(1);

    vector<vector<float>> output = network.feedForward(X);
    vector<float> Ypred = {output[0][0]};

    int rows = output.size();
    int columns = output.size();
    for(int i=0; i<rows; i++){
        for(int j=0; j<columns; j++){
            cout << output[i][j] << endl;
        }
    }

    float mse = network.mse(Y, Ypred);
    cout << mse << endl;

    // network.printNetwork();
}