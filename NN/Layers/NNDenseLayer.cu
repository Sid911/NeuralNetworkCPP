//
// Created by sid on 17/6/23.
//

#include <iostream>
#include "NNDenseLayer.cuh"

NNDenseLayer::NNDenseLayer(uint32_t _size, std::mt19937& _gen, bool _is_random) : input_size(_size), output_size(_size),
                                                              is_random(_is_random), gen(_gen) {
    values = make_shared<Eigen::VectorXf>();
    gen = mt19937(556);
}

NNDenseLayer::NNDenseLayer(uint32_t _input_size, uint32_t output_size, mt19937 &_gen, bool _is_random) : input_size(_input_size),
                                                                                          output_size(output_size),
                                                                                          is_random(_is_random),
                                                                                          gen(_gen) {
    values = make_shared<Eigen::VectorXf>();

}

void NNDenseLayer::allocate_layer(float min_value, float max_value) {
    weights.resize(input_size, output_size);
    biases.resize(output_size);
    weight_gradients.resize(input_size, output_size);
    bias_gradients.resize(output_size);
    values = std::make_shared<Eigen::VectorXf>(output_size);

    if (is_random) {
        // Generate random values for weights and biases
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

    // Set values to 0
    values->setZero();
    bias_gradients.setZero();
    weight_gradients.setZero();
}


shared_ptr<Eigen::VectorXf> NNDenseLayer::back_propagate(const std::shared_ptr<Eigen::VectorXf> &labels) {
    // Compute the error
    Eigen::VectorXf errors = (*values - *labels);

    cout << errors << "\n";

    // Compute the derivative of the activation function
    Eigen::VectorXf activationDerivative;
    if (false) {
        activationDerivative = (*values).unaryExpr(relu_derivative);
    } else {
        // Assuming the activation function is sigmoid
        activationDerivative = (*values).unaryExpr(sigmoid_derivative);
    }

    // Compute the delta
    Eigen::VectorXf delta = errors.cwiseProduct(activationDerivative);

    if(verbose_log)cout << "Delta : " << delta << "\n";

    // Compute the gradients of weights and biases
    weight_gradients = (delta * weights.transpose()).transpose();
    bias_gradients = delta;

    // Update the weights and biases
    weights -= learning_rate * weight_gradients;
    biases -= learning_rate * bias_gradients;
    if(verbose_log)
    {
        cout << "Weights : " << weights << "\tgradients : " << weight_gradients << endl;
        cout << "Biases : " << biases << "\tgradients : " << bias_gradients << endl;
    }

    return make_shared<Eigen::VectorXf>(bias_gradients);
}

shared_ptr<Eigen::VectorXf> NNDenseLayer::propagate(const shared_ptr<Eigen::VectorXf> &inp) {
    // z = L n-1 * weight + bias
    // a = activation fn (z)

    // Compute the linear combination z = L_{n-1} * weights + biases
    Eigen::VectorXf z = (weights * (*inp)) + biases;

    if(verbose_log)std::cout << "Weights : \n" << weights << "\nVector z : \n" << z << "\nBiases :" << biases << "\n";

    // Apply the activation function to compute the output a
    std::function<float(float)> activation = [this](float x) { return sigmoid_fn(x); };
    Eigen::VectorXf a = z.unaryExpr(activation);

    // Update the values in the layer
    *values = a;

    // Return the output vector a
    return values;
}


