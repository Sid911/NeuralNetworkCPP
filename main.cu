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

float sigmoid_fn(int x) {
    return 0.5 * (x / (1 + std::abs(x)) + 1);
}

float loss_function(float expected, float found) {
    return std::pow(found - expected, 2);
}

float calculate_loss(std::vector<std::vector<float>> &train_data, float w) {
    float total_cost = 0;
    for (auto &i: train_data) {
        auto x = i[0];
        float z = x * w;
        std::cout << "X : " << x << " W : " << w << " Z : " << z << "\n";
        float cost = loss_function(i[1], z);
        total_cost += cost;
    };
    total_cost /= (float) train_data.size();
    std::cout << "Total Cost: " << total_cost << "\n";
    return total_cost;
}


// Main program
int main() {
    vector<NNDenseLayer> layers = {
            NNDenseLayer(1, true),
            NNDenseLayer(1, true),
            NNDenseLayer(1, true)
    };
    NNSequentialModel model = NNSequentialModel(layers);
    shared_ptr<vector<float>> input (new vector<float>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}));
    vector<float> labels = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22};
    model.train(input, labels, 100);

//    vector<float> prediction = {50, 20};
//    model.predict(prediction);
}

//int main()
//{
//    std::srand(69); // use current time as seed for random generator
//    float w = (float) std::rand() / (float) RAND_MAX * 10;
//    std::cout << "W = " << w << "\n";
//    std::vector<std::vector<float>> train_data = {
//            {0, 0},
//            {1, 3},
//            {2, 6},
//            {3, 9},
//            {4, 12},
//            {5, 15},
//            {6, 18},
//    };
//
//    for (int n = 0; n < TRAIN_COUNT; n++)
//    {
//        float avg_d = 0;
//        for (auto &i: train_data) {
//            auto &x = i[0];
//            auto &y = i[1];
//            auto d = (float ) 2.0 *(x - y);
//            avg_d += d;
//        }
//        avg_d /= train_data.size();
//        w -= avg_d;
//        float total_loss = calculate_loss(train_data, w);
//    }
//
//    return 0;
//}