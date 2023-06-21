//
// Created by sid on 17/6/23.
//

#include <iostream>
#include "NNDenseLayer.cuh"

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
        const Eigen::MatrixXf &pre_w) {
    // Compute the delta
    Eigen::VectorXf delta = *pre_delta;
    delta = delta.cwiseProduct(z_vec->unaryExpr(relu_derivative));

    if (verbose_log) cout << "Delta : " << delta << "\n";

    // Update the weights and biases
    auto del_w = learning_rate * (delta);

    if (verbose_log)
        cout << "Prev W : " << weights << "\nPrev biases : " << biases << "\n";
    weights -= del_w;
    biases -= learning_rate * delta;

    if (verbose_log)
        cout << "Next W : " << weights << "\nNext biases : " << biases << "\n";

    Eigen::VectorXf next_delta_without_act_der = delta * weights.transpose();
    next_delta_without_act_der = next_delta_without_act_der.cwiseProduct(z_vec->unaryExpr(relu_derivative));

    return make_shared<Eigen::VectorXf>(next_delta_without_act_der);
}

shared_ptr<Eigen::VectorXf> NNDenseLayer::propagate(const shared_ptr<Eigen::VectorXf> &inp) {
    // Compute the linear combination z = I_{n} * weights + biases
    Eigen::VectorXf z = (weights * (*inp));
    if (verbose_log) cout << "Z size : " << z.rows() << " x " << z.cols() << endl;

    if (verbose_log)std::cout << "Weights : \t" << weights << "\nVector z : \t" << z << "\nBiases : " << biases << "\n";

    // Apply the activation function to compute the output a
    std::function<float(float)> activation = [](float x) { return relu(x); };
    Eigen::VectorXf a = z.unaryExpr(activation);
    this->activations = make_shared<Eigen::VectorXf>(a);

    // Update the z_vec in the layer
    *z_vec = a;

    cout << "A : " << a << "\n";

    // Return the output vector a
    return make_shared<Eigen::VectorXf>(a);
}


