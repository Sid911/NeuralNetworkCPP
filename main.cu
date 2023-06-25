#define EIGEN_NO_CUDA

#include <vector>
#include <iostream>
#include "NN/Models/NNSequentialModel.cuh"
#include "NN/Layers/NNInputLayer.cuh"
// Kernel
//__global__ void add_vectors(const double *a, const double *b, double *c) {
//    uint32_t id = blockDim.x * blockIdx.x + threadIdx.x;
//    if (id < N) c[id] = a[id] + b[id];
//}

#define N 20

using namespace Eigen;

void generate_linear(vector<shared_ptr<VectorXf>> &input, vector<shared_ptr<VectorXf>> &labels);

void generate_linear(MatrixXf &input, MatrixXf &labels) noexcept;

// Main program
int main() {
    mt19937 gen(0);

    vector<shared_ptr<NNLayer>> layers = {
            make_shared<NNInputLayer>(NNInputLayer(2, 2, gen, true)),
//            make_shared<NNDenseLayer>(NNDenseLayer(2, 2, gen, true)),
            make_shared<NNDenseLayer>(NNDenseLayer(2, 1, gen, true)),
    };
    NNSequentialModel model = NNSequentialModel(layers);

    MatrixXf input(N, 1);
    MatrixXf labels(N, 1);

//    std::vector<std::shared_ptr<VectorXf>> inputLabels;

    // Populate the input and labels vectors

    generate_linear(input, labels);
//    generate_gate(input, labels);

    model.train(input, labels, 1000);

//    auto pred_inp = 20.0f;
//    auto  prediction = make_shared<VectorXf>(VectorXf{{pred_inp}});
//    auto val = *model.predict(prediction);
//    std::cout << "Prediction : x = 20 then y = \n " << val << "\nLoss : "
//    << std::pow(val(0,0) - (pred_inp*3 + 2), 2);

    auto val = model.predict(input);
    std::cout << "Prediction : \n" << input << "\nOut : \n" << *val << std::endl;

}

void generate_linear(MatrixXf &input, MatrixXf &labels) noexcept {
    for (int i = 0; i < N; i++) {
        input(i, 0) = static_cast<float>(i);

        // equation here
        float y = (3.0f * (float) i) + 2.0f;

        labels(i, 0) = static_cast<float>(y);
    }
}

void generate_gate(MatrixXf &input, MatrixXf &labels) noexcept {
    // 0 , 0
    input(0, 0) = 0;
    input(0, 1) = 0;
    // 0 ,1
    input(1, 0) = 0;
    input(1, 1) = 1;
    // 1, 0
    input(2, 0) = 1;
    input(2, 1) = 0;
    // 1, 1
    input(3, 0) = 1;
    input(3, 1) = 1;

    labels(0, 0) = 0;
    labels(1, 0) = 0;
    labels(2, 0) = 0;
    labels(3, 0) = 1;

}

