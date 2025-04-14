#include "loss.hpp"

vector<vector<float>> mseDerivative(vector<vector<float>> yPred, vector<vector<float>> yTrue){
    
    vector<vector<float>> output (yPred.size(), vector<float>(yPred[0].size(), 0));

    for(size_t i = 0; i < yPred.size(); i++){
        for (size_t j = 0; j < yPred[0].size(); j++){
            output[i][j] = 2 * (yPred[i][j] - yTrue[i][j]);
        }
    }

    return output;

}

