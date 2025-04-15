#include "matrix_operations.hpp"

matrix matrixMultiplication(const matrix& a, const matrix& b){
    int rowsA = a.size();
    int columnsA = a[0].size();

    int rowsB = b.size();
    int columnsB = b[0].size();

    if(columnsA != rowsB){
        throw invalid_argument(
            "Matrix A has shape (" + to_string(rowsA) + "," + to_string(columnsA) +") and " +
            "matrix B has shape (" + to_string(rowsB) + "," + to_string(columnsB) +").\n" +
            "The number of columns of matrix A must match the number of rows of matrix B"
        );
    }

    matrix output(rowsA, vector<float>(columnsB, 0));

    for(int i=0; i < rowsA; i++)
        for(int j=0; j < columnsB; j++)
            for(int k=0; k < columnsA; k++) // columnsA = rowsB 
                output[i][j] += a[i][k] * b[k][j];
            
    return output;   
}


matrix elementWiseOperation(const matrix&  a, const matrix& b, float (*op)(float, float)){
    int rowsA = a.size();
    int columnsA = a[0].size();

    int rowsB = b.size();
    int columnsB = b[0].size();

    if(rowsA != rowsB || columnsA != columnsB){
        throw invalid_argument(
            "Matrix A has shape (" + to_string(rowsA) + "," + to_string(columnsA) +") and " +
            "matrix B has shape (" + to_string(rowsB) + "," + to_string(columnsB) +").\n" +
            "The matrices must have the same shape"
        );
    }

    matrix output(rowsA, vector<float>(columnsB, 0));

    for(int i=0; i<rowsA; i++)
        for(int j=0; j<columnsA; j++)
            output[i][j] = op(a[i][j], b[i][j]);
    
    return output;
}

matrix matrixAddition(const matrix& a, const matrix& b){
    auto op = [](float a, float b) {return a + b;};
    return elementWiseOperation(a, b, op);
}

matrix matrixSubtraction(const matrix& a, const matrix& b){
    auto op = [](float a, float b) {return a - b;};
    return elementWiseOperation(a, b, op);
}

matrix hadamardProduct(const matrix& a, const matrix& b){
    auto op = [](float a, float b) {return a * b;};
    return elementWiseOperation(a, b, op);
}

matrix transposeMatrix (const matrix& a){
    int rows = a.size();
    int columns = a[0].size();

    matrix transpose(columns, vector<float>(rows, 0));

    for(int i=0; i<rows; i++)
        for(int j=0; j<columns; j++)
            transpose[j][i] = a[i][j];

    return transpose;
}

matrix multiplicationByScalar(float scalar, const matrix &a){
    int rows = a.size();
    int columns = a[0].size();

    matrix output(rows, vector<float>(columns, 0));

    for(size_t i = 0; i < a.size(); i ++){
        for(size_t j = 0; j < a[0].size(); j++){
            output[i][j] = a[i][j] * scalar;
        }
    }

    return output;
}
