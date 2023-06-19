#include <iostream>
#include <vector>
#include "NN/Models/NNSequentialModel.cuh"

// Size of array
#define TRAIN_COUNT 10
#define N 1035264
// Kernel
__global__ void add_vectors(const double *a, const double *b, double *c) {
    uint32_t id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < N) c[id] = a[id] + b[id];
}

// Main program
int main() {
    mt19937 gen(500);
    vector<NNDenseLayer> layers = {
            NNDenseLayer(1, gen, true),
            NNDenseLayer(1, gen, true),
            NNDenseLayer(1, gen, true)
    };
    NNSequentialModel model = NNSequentialModel(layers);

    vector<shared_ptr<Eigen::VectorXf>> input;
    vector<shared_ptr<Eigen::VectorXf>> labels;

    std::vector<std::shared_ptr<Eigen::VectorXf>> inputLabels;

    // Populate the input and labels vectors
    for (int i = 0; i <= 11; ++i) {
        std::shared_ptr<Eigen::VectorXf> inputVector(new Eigen::VectorXf(1));
        (*inputVector)(0) = static_cast<float>(i);

        std::shared_ptr<Eigen::VectorXf> labelVector(new Eigen::VectorXf(1));
        (*labelVector)(0) = static_cast<float>(2 * i);

        input.push_back(inputVector);
        labels.push_back(labelVector);
    }
    model.train(input, labels, 5);

//    vector<float> prediction = {50, 20};
//    model.predict(prediction);
}