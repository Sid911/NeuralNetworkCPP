#define Ndebug

#include <vector>
#include <iostream>
#include "NN/Models/NNSequentialModel.cuh"
#include "NN/Layers/NNInputLayer.cuh"
// Kernel
//__global__ void add_vectors(const double *a, const double *b, double *c) {
//    uint32_t id = blockDim.x * blockIdx.x + threadIdx.x;
//    if (id < N) c[id] = a[id] + b[id];
//}

#define N 4

using namespace Eigen;

void generate_linear(MatrixXf &input, MatrixXf &labels) noexcept;
void generate_gate(MatrixXf &input, MatrixXf &labels) noexcept;

// Main program
int main() {
    mt19937 gen(time(nullptr));
    MatrixXf layer1m({
        {0.185689, 0.688531 },
        {0.715891, 0.694503 }
    });

    MatrixXf layer2m({
        {-0.404931, -0.886574}
    });

    VectorXf layer1b({
        { 0.247127, -0.231237}
    });

    VectorXf layer2b({
        { -0.454687 }
    });
    vector<shared_ptr<NNLayer>> layers = {
            make_shared<NNInputLayer>(NNInputLayer(2, 2, gen, true)),
            make_shared<NNDenseLayer>(NNDenseLayer(2, 2, gen, true)),
            make_shared<NNDenseLayer>(NNDenseLayer(2, 1, gen, true)),
    };
    NNSequentialModel model = NNSequentialModel(layers);

    MatrixXf input(N, 2);
    MatrixXf labels(N, 1);

    // Populate the input and labels vectors

//    generate_linear(input, labels);
    generate_gate(input, labels);

    model.train(input, labels, 500);

    auto val = model.predict(input);


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
    labels(1, 0) = 1;
    labels(2, 0) = 1;
    labels(3, 0) = 0;

}

