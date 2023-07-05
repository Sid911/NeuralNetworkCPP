//#define NNdebug

#include "pch.cuh"
#include "NN/Utils/Logger.cuh"
#include "NN/Models/NNSequentialModel.cuh"
#include "NN/Layers/NNInputLayer.cuh"
#include "DataLoaders/ImageData.h"

#define N 4

using namespace Eigen;

void generate_linear(MatrixXf &input, MatrixXf &labels) noexcept;
void generate_gate(MatrixXf &input, MatrixXf &labels) noexcept;


int main() {
    mt19937 gen(time(nullptr));
    const float lr = 2;
    vector<shared_ptr<NNLayer>> layers = {
            make_shared<NNInputLayer>(NNInputLayer(2, 2, gen, true)),
//            make_shared<NNDenseLayer>(NNDenseLayer(2, 3, gen, true)),
            make_shared<NNDenseLayer>(NNDenseLayer(2, 1, gen, true, lr)),
    };
    NNSequentialModel model = NNSequentialModel(layers);

    MatrixXf input(N, 2);
    MatrixXf labels(N, 1);

    // Populate the input and trainingLabels vectors

//    generate_linear(input, trainingLabels);
    generate_gate(input, labels);

    model.train(input, labels, 500);

    auto val = model.predict(input);

    return 0;
};

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
    labels(1, 0) = 1;
    labels(2, 0) = 1;
    labels(3, 0) = 1;

}

