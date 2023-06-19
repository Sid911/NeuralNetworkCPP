//
// Created by sid on 17/6/23.
//
#pragma once
#ifndef NNCPP_NNDENSELAYER_CUH
#define NNCPP_NNDENSELAYER_CUH

#define assertm(exp, msg) assert(((void)msg, exp))

#include <cstdint>
#include <vector>
#include <random>
#include <memory>
#include <cassert>
#include "Eigen/Dense"

using namespace std;


class NNDenseLayer {
public:
    uint32_t input_size, output_size;

    float learning_rate = 0.001f;
    bool is_random;
    bool verbose_log = false;

    Eigen::MatrixXf weights;
    Eigen::VectorXf  biases;

    explicit NNDenseLayer(uint32_t _size, mt19937& _gen, bool _is_random = true);

    NNDenseLayer(uint32_t _input_size, uint32_t output_size, mt19937& _gen, bool _is_random);

    void allocate_layer(float = 0.0f, float = 1.0f );

    shared_ptr<Eigen::VectorXf > propagate(const shared_ptr<Eigen::VectorXf >& inp);

    shared_ptr<Eigen::VectorXf > back_propagate(const std::shared_ptr<Eigen::VectorXf >& z_vec);

private:
    /*
     * The weights of r rows (each row is for 1 neuron in the layer)
     * The column c represents weights per input.
     * matrix being input_size * output_size
     */
    Eigen::MatrixXf weight_gradients;
    Eigen::VectorXf bias_gradients;
    std::mt19937& gen;

    shared_ptr<Eigen::VectorXf > values;

    shared_ptr<Eigen::VectorXf > compute_error(shared_ptr<Eigen::VectorXf> labels){
        shared_ptr<Eigen::VectorXf > errors(new Eigen::VectorXf (output_size));

        return errors;
    };

    inline float sigmoid_fn(float x) {
        return 0.5 * (x / (1 + std::abs(x)) + 1);
    }
    inline float relu(float x){
        return max(0.0f,x);
    }
    constexpr static const auto relu_derivative = [](float x) {
        return (x > 0) ? 1.0f : 0.0f ;};
    };

    constexpr static const auto sigmoid_derivative = [](float x) {
        float sigmoid = 1.0f / (1.0f + std::exp(-x));
        return sigmoid * (1.0f - sigmoid);
    };


#endif //NNCPP_NNDENSELAYER_CUH
