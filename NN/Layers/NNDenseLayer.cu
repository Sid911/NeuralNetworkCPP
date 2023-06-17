//
// Created by sid on 17/6/23.
//

#include <iostream>
#include "NNDenseLayer.cuh"

template<typename Tp>
NNDenseLayer<Tp>::NNDenseLayer(uint32_t _size, bool _is_random): input_size(_size), output_size(_size),
                                                                 is_random(_is_random) {

}

template<typename Tp>
NNDenseLayer<Tp>::NNDenseLayer(uint32_t _input_size, uint32_t output_size, bool _is_random): input_size(_input_size),
                                                                                             output_size(output_size),
                                                                                             is_random(_is_random) {

}

template<typename Tp>
void NNDenseLayer<Tp>::allocate_layer() {
    weights.resize(output_size, vector<float>(input_size));
    biases.resize(output_size);
    values = make_shared<vector<float>>();
    (*values).resize(output_size);

    if (is_random) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0, 1.0);

        for (uint32_t i = 0; i < output_size; ++i) {
            for (uint32_t j = 0; j < input_size; ++j) {
                weights[i][j] = dis(gen);
            }
            biases[i] = dis(gen);
            (*values)[i] = dis(gen);
        }
    } else {
        for (uint32_t i = 0; i < output_size; ++i) {
            for (uint32_t j = 0; j < input_size; ++j) {
                weights[i][j] = 0.0;
            }
            biases[i] = 0.0;
            (*values)[i] = 0.0;
        }
    }
}

template<typename Tp>
shared_ptr<vector<Tp>> NNDenseLayer<Tp>::back_propagate(std::shared_ptr<vector<float>> z_vec) {
    return shared_ptr<vector<Tp>>();
}

template<typename Tp>
const vector<Tp> &NNDenseLayer<Tp>::propagate(const vector<Tp> &inp) {
    // z = L n-1 * weight + bias
    // a = activation fn (z)
    for (auto i = 0; i < output_size; i++) {
        float z = biases[i];
        for (auto j = 0; j < input_size; j++) {
            z += inp[j] * weights[i][j];
        }
        auto a = sigmoid_fn(z);
        (*values)[i] = a;
        std::cout << a << " ";
    }
    return values;
}