#include "matrix_operations.hpp"

vector<vector<float>> matrixMultiplication(vector<vector<float>> a, vector<vector<float>> b){
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

    vector<vector<float>> output(rowsA, vector<float>(columnsB, 0));

    for(int i=0; i < rowsA; i++)
        for(int j=0; j < columnsB; j++)
            for(int k=0; k < columnsA; k++) // columnsA = rowsB 
                output[i][j] += a[i][k] * b[k][j];
            
    return output;   
}


vector<vector<float>> elementWiseOperation(vector<vector<float>> a, vector<vector<float>> b, float (*op)(float, float)){
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

    vector<vector<float>> output(rowsA, vector<float>(columnsB, 0));

    for(int i=0; i<rowsA; i++)
        for(int j=0; j<columnsA; j++)
            output[i][j] = op(a[i][j], b[i][j]);
    
    return output;
}

vector<vector<float>> matrixAddition(vector<vector<float>> a, vector<vector<float>> b){
    auto op = [](float a, float b) {return a + b;};
    return elementWiseOperation(a, b, op);
}

vector<vector<float>> matrixSubtraction(vector<vector<float>> a, vector<vector<float>> b){
    auto op = [](float a, float b) {return a - b;};
    return elementWiseOperation(a, b, op);
}

vector<vector<float>> hadamardProduct(vector<vector<float>> a, vector<vector<float>> b){
    auto op = [](float a, float b) {return a * b;};
    return elementWiseOperation(a, b, op);
}

vector<vector<float>> transposeMatrix (vector<vector<float>> a){
    int rows = a.size();
    int columns = a[0].size();

    vector<vector<float>> transpose(columns, vector<float>(rows, 0));

    for(int i=0; i<rows; i++)
        for(int j=0; j<columns; j++)
            transpose[j][i] = a[i][j];

    return transpose;
}

vector<vector<float>> multiplicationByScalar(float scalar, vector<vector<float>> a){
    for(size_t i = 0; i < a.size(); i ++){
        for(size_t j = 0; j < a[0].size(); j++){
            a[i][j] *= scalar;
        }
    }

    return a;
}
