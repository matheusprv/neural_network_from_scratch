#ifndef MATRIX_OPERATIONS_HPP
#define MATRIX_OPERATIONS_HPP

#include <vector>
#include <stdexcept>

using namespace std;
 
#define matrix vector<vector<float>>

matrix matrixMultiplication(const matrix& a, const matrix& b);

matrix elementWiseOperation(const matrix& a, const matrix& b, float (*op)(float, float));

matrix matrixAddition(const matrix& a, const matrix& b);

matrix matrixSubtraction(const matrix& a, const matrix& b);
    
matrix hadamardProduct(const matrix& a, const matrix& b);
    
matrix transposeMatrix(const matrix& a);

matrix multiplicationByScalar(float scalar, const matrix& a);


#endif