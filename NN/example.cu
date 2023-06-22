//
// Created by sid on 21/6/23.
//
#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Core>

using namespace Eigen;

class NN {
private:
    std::vector<int> topology; // Topology of the neural network
    std::vector<MatrixXd> weights; // Weights of the neural network
    std::vector<VectorXd> biases; // Biases of the neural network
    std::vector<VectorXd> activations; // Activations of each layer

public:
    explicit NN(const std::vector<int>& topology) : topology(topology) {
        Eigen::setNbThreads(8);
        srand(time(nullptr));
        // Initialize the weights and biases with random values
        for (int i = 1; i < topology.size(); i++) {
            int currentLayerSize = topology[i];
            int prevLayerSize = topology[i - 1];

            MatrixXd weightMatrix = MatrixXd::Random(currentLayerSize, prevLayerSize);
            VectorXd biasVector = VectorXd::Random(currentLayerSize);

            weights.push_back(weightMatrix);
            biases.push_back(biasVector);
        }
    }

    VectorXd forward(const VectorXd& input) {
        activations.clear();
        activations.push_back(input);

        for (int i = 0; i < topology.size() - 1; i++) {
            VectorXd z = weights[i] * activations[i] + biases[i];
            VectorXd a = z.unaryExpr([](double x) { return std::max(0.0, x); });

            activations.push_back(a);
        }

        return activations.back();
    }

    void backward(const VectorXd& target, double learningRate) {
        std::vector<VectorXd> deltas(topology.size() - 1);

        // Calculate the delta for the output layer
        deltas.back() = activations.back() - target;

        // Backpropagate the deltas
        for (int i = topology.size() - 2; i > 0; i--) {
            VectorXd z = weights[i].transpose() * deltas[i];
            VectorXd derivative = activations[i].unaryExpr([](double x) { return x > 0.0 ? 1.0 : 0.0; });
            deltas[i - 1] = z.cwiseProduct(derivative);
        }

        // Update the weights and biases
        for (int i = 0; i < topology.size() - 1; i++) {
            weights[i] -= learningRate * deltas[i] * activations[i].transpose();
            biases[i] -= learningRate * deltas[i];
        }
    }
    void train(MatrixXd &trainingData, MatrixXd &targetValues, int numEpochs, double learningRate) noexcept {
        for (int epoch = 0; epoch < numEpochs; epoch++) {
            double totalLoss = 0.0;

            for (int i = 0; i < trainingData.rows(); i++) {
                // Forward propagation
                VectorXd input = trainingData.row(i);
                VectorXd output = forward(input);

                // Backward propagation
                VectorXd target = targetValues.row(i);
                backward(target, learningRate);

                // Calculate loss
                double loss = (output - target).squaredNorm();
                totalLoss += loss;
            }

            // Print average loss for the epoch
            double avgLoss = totalLoss / trainingData.rows();
            std::cout << "Epoch: " << epoch << ", Average Loss: " << avgLoss << std::endl;
        }
    }

    void test( MatrixXd &trainingData, MatrixXd &targetValues) noexcept {// Test the trained network
        std::cout << "Testing the trained network:" << std::endl;

        for (int i = 0; i < trainingData.rows(); i++) {
            VectorXd input = trainingData.row(i);
            VectorXd output = forward(input);

            std::cout << "Input: " << input.transpose() << ", Output: " << output;
            std::cout << " Expected: " << targetValues.row(i).transpose() << std::endl;
        }
    }
};





int main2() {
    // Define the topology of the neural network
    std::vector<int> topology = {1,2, 1};
    constexpr auto N = 50;

    // Create an instance of the NN class
    NN network(topology);

    // Training data
    MatrixXd trainingData(N+1, 1);

    // Target values
    MatrixXd targetValues(N +1 , 1);

    for (int x = 0; x <= N; x++) {
        int y = (x * 2 + 1) + (3*x);
        trainingData(x, 0) = x;
        targetValues(x, 0) = y;
    }

    // Training loop
    int numEpochs = 1000;
    double learningRate = 0.00001;

    network.train(trainingData, targetValues, numEpochs, learningRate);
    network.test(trainingData, targetValues);

    return 0;
}
