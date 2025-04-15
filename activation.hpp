#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "vector"
using namespace std;

vector<vector<float>> ReLU(vector<vector<float>> z);

vector<vector<float>> ReLUDerivative(vector<vector<float>> z);

vector<vector<float>> linearDerivative(vector<vector<float>> z);

#endif