//
// Created by sid on 17/6/23.
//

#include <iostream>
#include "NNDenseLayer.cuh"

void NNDenseLayer::allocate_layer(float min_value, float max_value) {
    weights.resize(output_size, input_size);
    biases.resize(output_size);
    z_vec = std::make_shared<Eigen::VectorXf>(output_size);

    if (is_random) {
        // Generate random z_vec for weights and biases
        std::uniform_real_distribution<float> dis(min_value, max_value);
        std::uniform_real_distribution<float> b_dis(min_value, max_value);

        for (uint32_t i = 0; i < output_size; i++) {
            for (uint32_t j = 0; j < input_size; j++) {
                weights(i, j) = dis(gen);
            }
        }

        for (uint32_t i = 0; i < output_size; i++) {
            biases(i) = b_dis(gen);
        }
    } else {
        // Set weights and biases to 0
//        weights.setZero();
//        biases.setZero();
        weights(0,0) = 0.185689;
        weights(0,1) = 0.618044;
        biases(0) = 0.7158912399;
    }

    // Set z_vec to 0
    z_vec->setZero();

}


shared_ptr<Eigen::VectorXf> NNDenseLayer::back_propagate(
        const std::shared_ptr<Eigen::VectorXf> &target) {
    // Compute the delta
    delta = target;

    Eigen::VectorXf z = weights.transpose() * *delta;

    if (verbose_log) cout << "Delta : " << *delta << "\n";

    Eigen::VectorXf derivative = activations->unaryExpr(relu_derivative);
    auto next_target = make_shared<Eigen::VectorXf>(z.cwiseProduct(derivative));
    return next_target;
}

shared_ptr<Eigen::VectorXf> NNDenseLayer::propagate(const shared_ptr<Eigen::VectorXf> &pre_act) {
    // Compute the linear combination z = I_{n} * weights + biases
    Eigen::VectorXf z = (weights * (*pre_act)) + biases; // returns very different
//    Eigen::VectorXf z = (weights * (*pre_act)); // Without bias the result is correct
    if (verbose_log) cout << "Z size : " << z.rows() << " x " << z.cols() << endl;

    if (verbose_log)std::cout << "Weights : \t" << weights << "\nVector z : \t" << z << "\nBiases : " << biases << "\n";

    // Apply the activation function to compute the output a
    std::function<float(float)> activation = [](float x) { return relu(x); };
    Eigen::VectorXf a = z.unaryExpr(activation);
    this->activations = pre_act;

    // Update the z_vec in the layer
    *z_vec = z;

    if (verbose_log)cout << "A : " << a << "\n";

    // Return the output vector a
    return make_shared<Eigen::VectorXf>(a);
}

void NNDenseLayer::update_parameters() {
    if (verbose_log) {
        cout << "Prev W : \n" << weights << "\nPrev B : \n" << biases << "\n";
    }
    weights -= learning_rate * *delta * activations->transpose();
    biases -= learning_rate * *delta;
    if (verbose_log) {
        cout << "Next W : \n" << weights << "\nNext B : \n" << biases << "\n";
    }
}


