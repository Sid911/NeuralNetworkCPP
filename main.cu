#include <vector>
#include <iostream>
#include "NN/Models/NNSequentialModel.cuh"
#include "NN/Layers/NNInputLayer.cuh"

// Kernel
//__global__ void add_vectors(const double *a, const double *b, double *c) {
//    uint32_t id = blockDim.x * blockIdx.x + threadIdx.x;
//    if (id < N) c[id] = a[id] + b[id];
//}

// Main program
int main() {
    mt19937 gen(500);
    vector<shared_ptr<NNLayer>> layers = {
            make_shared<NNInputLayer>(NNInputLayer(1,1, gen, true)),
            make_shared<NNDenseLayer>(NNDenseLayer(1, gen, true)),
//            make_shared<NNDenseLayer>(NNDenseLayer(1, gen, true)),
    };
    NNSequentialModel model = NNSequentialModel(layers);

    vector<shared_ptr<Eigen::VectorXf>> input;
    vector<shared_ptr<Eigen::VectorXf>> labels;

    std::vector<std::shared_ptr<Eigen::VectorXf>> inputLabels;

    // Populate the input and labels vectors
    for (int i = 1; i <= 11; ++i) {
        std::shared_ptr<Eigen::VectorXf> inputVector(new Eigen::VectorXf(1));
        (*inputVector)(0) = static_cast<float>(i);

        std::shared_ptr<Eigen::VectorXf> labelVector(new Eigen::VectorXf(1));
        (*labelVector)(0) = static_cast<float>((3 * i));

        input.push_back(inputVector);
        labels.push_back(labelVector);
    }
    model.train(input, labels, 5);

    auto  prediction = make_shared<Eigen::VectorXf>(Eigen::VectorXf{{20.0f}});
    auto val = *model.predict(prediction);
    std::cout << "Prediction : x = 19 then y = \n " << val;
}