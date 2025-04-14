using namespace std;


#include "network.hpp"

int main() {

    /*
        5 neuronios de entrada
        3
        3
        1

        Y_true = (outputs, batches) = (1, batches)
    */

    input X = {
        {1}, 
        {2}, 
    };

    output Y = {
        {3.0}
    };


    // m: inputs - n: outputs -> A matriz de pesos deve ser (n,m) Uma linha para cada neuronio

    return 0;
}