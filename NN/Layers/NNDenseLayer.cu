//
// Created by sid on 17/6/23.
//

#include <iostream>
#include "NNDenseLayer.cuh"

NNDenseLayer::NNDenseLayer(uint32_t _size, bool _is_random): input_size(_size), output_size(_size),
                                                             is_random(_is_random) {
    values = make_shared<vector<float>>();
}

NNDenseLayer::NNDenseLayer(uint32_t _input_size, uint32_t output_size, bool _is_random): input_size(_input_size),
                                                                                         output_size(output_size),
                                                                                         is_random(_is_random) {

}

void NNDenseLayer::allocate_layer(float range_min, float range_max) {
    weights.resize(output_size, vector<float>(input_size));
    biases.resize(3);
    values->resize(output_size);

    if (is_random) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(range_min, range_max);

        for (uint32_t i = 0; i < output_size; ++i) {
            for (uint32_t j = 0; j < input_size; ++j) {
                weights[i][j] = dis(gen);
            }
            biases[i] = dis(gen);
            values->at(i) = dis(gen);
        }
    } else {
        for (uint32_t i = 0; i < output_size; ++i) {
            for (uint32_t j = 0; j < input_size; ++j) {
                weights[i][j] = 0.0;
            }
            biases[i] = 0.0;
            values->at(i) = 0.0;
        }
    }
}

shared_ptr<vector<float>> NNDenseLayer::back_propagate(std::shared_ptr<vector<float>> labels) {
    // here labels is the truth/values vector from next layer
    // here values is the current values vector is the values in the neuron
    // weights 2d vector is the weights connected to previous node

}

shared_ptr<vector<float>> NNDenseLayer::propagate(shared_ptr<vector<float>> inp) {
    // z = L n-1 * weight + bias
    // a = activation fn (z)
    for (auto i = 0; i < output_size; i++) {
        float z = biases[i];
        for (auto j = 0; j < input_size; j++) {
            z += inp->at(j) * weights[i][j];
        }
        auto a = relu(z + biases[i]);
        values->at(i) = a;
        std::cout << a << " ";
    }
    return values;
}


