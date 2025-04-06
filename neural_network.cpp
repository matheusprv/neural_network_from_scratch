#include <iostream>
#include <vector>
#include <stdexcept>

using namespace std;

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

vector<vector<float>> addBias(vector<vector<float>> a, vector<float> b){
    int rowsA = a.size();
    int columnsA = a[0].size();

    int columnsB = b.size();

    if(rowsA != 1 || columnsA != columnsB){
        throw invalid_argument(
            "Matrix A has shape (" + to_string(rowsA) + "," + to_string(columnsA) +") and " +
            "matrix B has shape (" + to_string(1) + "," + to_string(columnsB) +").\n" +
            "The matrices must have the same shape"
        );
    }

    vector<vector<float>> output(rowsA, vector<float>(columnsB, 0));

    for(int i=0; i<rowsA; i++)
        for(int j=0; j<columnsA; j++)
            output[i][j] = a[i][j] + b[j];
    
    return output;
}

vector<vector<float>> ReLU(vector<vector<float>> z){
    int rows = z.size();
    int columns = z[0].size();

    for(int i=0; i<rows; i++)
        for(int j=0; j<columns; j++)
            if(z[i][j] < 0)
                z[i][j] = 0;

    return z;
}

vector<vector<float>> feedLayer(vector<vector<float>> X, vector<vector<float>> W, vector<float> b){    
    vector<vector<float>> z = matrixMultiplication(X, W);
    z = addBias(z, b);
    vector<vector<float>> a = ReLU(z);

    return a;
}


int main() {

    vector<vector<float>> X = {
        {-0.06639924, 1.82608733, -0.48022676, -0.83469193, -0.86759439}
    };

    vector<vector<float>> W1 = {
        {-0.5856736 , -0.34722853, -0.07142764},
        {-0.83059484,  0.84327275,  0.69788927},
        {-0.36352283,  0.4653539 , -0.68444496},
        { 0.854593  ,  0.10911006, -0.7500073 },
        { 0.6249593 , -0.03334123,  0.83828694}
    };     
    vector<float> b1(W1[0].size(), 0);
    
    vector<vector<float>> W2 = {
        {-0.7844622 ,  0.92668843,  0.01148081},
        { 0.9856701 ,  0.17767239,  0.55978155},
        { 0.8170264 ,  0.88097   ,  0.8334906 }
    };     
    vector<float> b2(W2[0].size(), 0);

    vector<vector<float>> W3 = {
        { 0.5403869}, {-0.9536976},  {0.6031604}
    };     
    vector<float> b3(W3[0].size(), 0);

    cout << "layer 1" << endl;
    vector<vector<float>> layer1 = feedLayer(X, W1, b1);
    cout << "layer 2" << endl;
    vector<vector<float>> layer2 = feedLayer(layer1, W2, b2);
    cout << "layer 3" << endl;
    vector<vector<float>> layer3 = feedLayer(layer2, W3, b3);

    int rows = layer3.size();
    int columns = layer3.size();
    for(int i=0; i<rows; i++){
        for(int j=0; j<columns; j++){
            cout << layer3[i][j];
        }
    }
}