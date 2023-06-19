//
// Created by sid on 17/6/23.
//

#include <iostream>
#include "NNDenseLayer.cuh"

NNDenseLayer::NNDenseLayer(uint32_t _size, bool _is_random): input_size(_size), output_size(_size),
                                                             is_random(_is_random) {
    values = make_shared<Eigen::VectorXf >();
}

NNDenseLayer::NNDenseLayer(uint32_t _input_size, uint32_t output_size, bool _is_random): input_size(_input_size),
                                                                                         output_size(output_size),
                                                                                         is_random(_is_random) {

}

void NNDenseLayer::allocate_layer(float min_value, float max_value) {
    weights.resize(input_size, output_size);
    biases.resize(output_size);
    values = std::make_shared<Eigen::VectorXf>(output_size);

    if (is_random) {
        // Generate random values for weights and biases
        weights = Eigen::MatrixXf::Random(input_size, output_size);
        weights = (weights.array() * (max_value - min_value) + (max_value + min_value)) / 2.0;

        biases = Eigen::VectorXf::Random(output_size);
        biases = (biases.array() * (max_value - min_value) + (max_value + min_value)) / 2.0;
    } else {
        // Set weights and biases to 0
        weights.setZero();
        biases.setZero();
    }

    // Set values to 0
    values->setZero();
}


shared_ptr<Eigen::VectorXf > NNDenseLayer::back_propagate(std::shared_ptr<Eigen::VectorXf > labels) {
    // here labels is the truth/values vector from next layer
    // here values is the current values vector is the values in the neuron
    // weights 2d vector is the weights connected to previous node

}

shared_ptr<Eigen::VectorXf > NNDenseLayer::propagate(shared_ptr<Eigen::VectorXf > inp) {
    // z = L n-1 * weight + bias
    // a = activation fn (z)


}


