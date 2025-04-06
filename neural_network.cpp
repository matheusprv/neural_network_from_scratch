#include <iostream>
#include <vector>
#include <stdexcept>

using namespace std;

#include "network.hpp"


int main() {

    vector<vector<float>> X = {
        {-0.96639924, 5.82608733, -0.68022676, -7.83469193, -0.86759439}
    };

    Network network(5);
    network.addLayer(3);
    network.addLayer(3);
    network.addLayer(1);

    vector<vector<float>> output = network.feedForward(X);
    
    int rows = output.size();
    int columns = output.size();
    for(int i=0; i<rows; i++){
        for(int j=0; j<columns; j++){
            cout << output[i][j] << endl;
        }
    }
}