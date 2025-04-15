#ifndef LOSS_H
#define LOSS_H

#include <vector>
#include <stdexcept>
using namespace std;

float squaredError(vector<vector<float>> yPred, vector<vector<float>> yTrue);

vector<vector<float>> seDerivative(vector<vector<float>> yPred, vector<vector<float>> yTrue);

#endif