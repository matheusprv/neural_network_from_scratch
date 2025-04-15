using namespace std;

#include "csv.hpp"
#include "network.hpp"

int main() {


    auto [X, Y] = readCSV("dataset/mnist_test.csv");
    

    Network network(784);
    network.addLayer(32, "relu");
    network.addLayer(32, "relu");
    network.addLayer(1);

    network.SGD(X, Y, 10);

    return 0;
}