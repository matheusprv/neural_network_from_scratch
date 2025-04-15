using namespace std;

#include "csv.hpp"
#include "network.hpp"

int main() {

    auto [X, Y] = readCSV("dataset/mnist_train.csv");
    

    Network network(784);
    network.addLayer(32, "relu");
    network.addLayer(32, "relu");
    network.addLayer(16, "relu");
    network.addLayer(1);

    network.SGD(X, Y, 10, 0.001f);

    cout << endl;
    auto [X_test, Y_test] = readCSV("dataset/mnist_test.csv");
    network.evaluation(X_test, Y_test);

    return 0;
}