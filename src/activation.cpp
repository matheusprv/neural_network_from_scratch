#include "activation.hpp"

vector<vector<float>> ReLU(vector<vector<float>> z){
    int rows = z.size();
    int columns = z[0].size();

    for(int i=0; i<rows; i++)
        for(int j=0; j<columns; j++)
            if(z[i][j] < 0)
                z[i][j] = 0;

    return z;
}

vector<vector<float>> ReLUDerivative(vector<vector<float>> z){
    int rows = z.size();
    int columns = z[0].size();

    for(int i=0; i<rows; i++)
        for(int j=0; j<columns; j++)
            z[i][j] = z[i][j] >= 0; // 1 if true, 0 otherwise
    return z;
}

vector<vector<float>> linearDerivative(vector<vector<float>> z){
    int rows = z.size();
    int columns = z[0].size();

    for(int i=0; i<rows; i++)
        for(int j=0; j<columns; j++)
            z[i][j] = 1; 
    return z;
}