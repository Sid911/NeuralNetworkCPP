//
// Created by sid on 17/6/23.
//

#include <iostream>
#include "NNDenseLayer.cuh"

NNDenseLayer::NNDenseLayer(uint32_t _size, std::mt19937 &_gen, bool _is_random) : input_size(_size), output_size(_size),
                                                                                  is_random(_is_random), gen(_gen) {
    z_vec = make_shared<Eigen::VectorXf>();
    gen = mt19937(556);
}

NNDenseLayer::NNDenseLayer(uint32_t _input_size, uint32_t output_size, mt19937 &_gen, bool _is_random) : input_size(
        _input_size),
                                                                                                         output_size(
                                                                                                                 output_size),
                                                                                                         is_random(
                                                                                                                 _is_random),
                                                                                                         gen(_gen) {
    z_vec = make_shared<Eigen::VectorXf>();

}

void NNDenseLayer::allocate_layer(float min_value, float max_value) {
    weights.resize(input_size, output_size);
    biases.resize(output_size);
    z_vec = std::make_shared<Eigen::VectorXf>(output_size);

    if (is_random) {
        // Generate random z_vec for weights and biases
        std::uniform_real_distribution<float> dis(min_value, max_value);
        std::uniform_real_distribution<float> b_dis(-abs(max_value), abs(max_value));

        for (uint32_t i = 0; i < input_size; i++) {
            for (uint32_t j = 0; j < output_size; j++) {
                weights(i, j) = dis(gen);
            }
        }

        for (uint32_t i = 0; i < output_size; i++) {
            biases(i) = b_dis(gen);
        }
    } else {
        // Set weights and biases to 0
        weights.setZero();
        biases.setZero();
    }

    // Set z_vec to 0
    z_vec->setZero();

}


shared_ptr<Eigen::VectorXf> NNDenseLayer::back_propagate(
        const std::shared_ptr<Eigen::VectorXf> &pre_delta,
        const std::shared_ptr<Eigen::VectorXf> &next_act) {
    // Compute the delta
    Eigen::VectorXf delta = *pre_delta * z_vec->unaryExpr(relu_derivative);

    if (verbose_log)cout << "Delta : " << delta << "\n";

    // Update the weights and biases
    auto del_w = learning_rate * (delta.dot(*next_act));
    weights -= weights * del_w;
    biases -= learning_rate * delta;

    auto next_delta_without_act_der = delta * weights.transpose();

    return make_shared<Eigen::VectorXf>(next_delta_without_act_der);
}

shared_ptr<Eigen::VectorXf> NNDenseLayer::propagate(const shared_ptr<Eigen::VectorXf> &inp) {
    // z = I_n * weight + bias
    // a = activation fn (z)

    // Compute the linear combination z = I_{n} * weights + biases
    Eigen::VectorXf z = (weights * (*inp)) + biases;
    if (verbose_log) cout << "Z size : " << z.rows() << " x " << z.cols() << endl;

    if (verbose_log)std::cout << "Weights : \t" << weights << "\nVector z : \t" << z << "\nBiases :" << biases << "\n";

    // Apply the activation function to compute the output a
    std::function<float(float)> activation = [](float x) { return relu(x); };
    shared_ptr<Eigen::VectorXf> a = make_shared<Eigen::VectorXf>(z.unaryExpr(activation));
    this->activations = a;

    // Update the z_vec in the layer
    *z_vec = z;

    cout << "A : " << *a << "\n";

    // Return the output vector a
    return a;
}


