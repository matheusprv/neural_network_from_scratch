# Neural Network from Scratch using C++

This project implements a simple feedforward neural network entirely from scratch in C++, using only standard libraries. It includes core concepts such as forward propagation, backpropagation, and training via Stochastic Gradient Descent (SGD).

The model supports ReLU and Linear activation functions, which are implemented alongside their derivatives in [activation.cpp](./src/activation.cpp).

The following code, which is also present in  [main.cpp](./src/main.cpp), creates a neural network with three hidden layers with ReLU activation and a linear output layer.

````
Network network(784);

network.addLayer(32, "relu");
network.addLayer(32, "relu");
network.addLayer(16, "relu");
network.addLayer(1);
````

After training for 10 epochs with a learning rate of 0.001 and evaluating, the following results were achieved on the [MNIST dataset](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv?resource=download&select=mnist_train.csv).

````
Starting epoch 1
mse: 3.34658
Starting epoch 2
mse: 1.56991
Starting epoch 3
mse: 1.25968
Starting epoch 4
mse: 1.08958
Starting epoch 5
mse: 0.971769
Starting epoch 6
mse: 0.881159
Starting epoch 7
mse: 0.811127
Starting epoch 8
mse: 0.761966
Starting epoch 9
mse: 0.717372
Starting epoch 10
mse: 0.674534

Final loss: 0.891553
````