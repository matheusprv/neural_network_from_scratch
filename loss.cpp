#include "loss.hpp"



float squaredError(vector<vector<float>> yPred, vector<vector<float>> yTrue){
    float summation = 0;
    float temp = 0.0;
    for(size_t i = 0; i < yTrue.size(); i++){
        for(size_t j = 0; j < yTrue[0].size(); j++){
            temp = yTrue[i][j] - yPred[i][j];
            summation += temp*temp;
        }
    }
    return summation;
}

vector<vector<float>> seDerivative(vector<vector<float>> yPred, vector<vector<float>> yTrue){
    
    vector<vector<float>> output (yPred.size(), vector<float>(yPred[0].size(), 0));

    for(size_t i = 0; i < yPred.size(); i++){
        for (size_t j = 0; j < yPred[0].size(); j++){
            output[i][j] = 2 * (yPred[i][j] - yTrue[i][j]);
        }
    }

    return output;

}

