#ifndef MATRIX_OPERATIONS_HPP
#define MATRIX_OPERATIONS_HPP

#include <vector>
#include <stdexcept>

using namespace std;
 

vector<vector<float>> matrixMultiplication(vector<vector<float>> a, vector<vector<float>> b);

vector<vector<float>> elementWiseOperation(vector<vector<float>> a, vector<vector<float>> b, float (*op)(float, float));

vector<vector<float>> matrixAddition(vector<vector<float>> a, vector<vector<float>> b);

vector<vector<float>> matrixSubtraction(vector<vector<float>> a, vector<vector<float>> b);
    
vector<vector<float>> hadamardProduct(vector<vector<float>> a, vector<vector<float>> b);
    
vector<vector<float>> transposeMatrix(vector<vector<float>> a);

vector<vector<float>> multiplicationByScalar(float scalar, vector<vector<float>> a);


#endif