//
// Created by sid on 17/6/23.
//
#pragma once
#ifndef NNCPP_NNDENSELAYER_CUH
#define NNCPP_NNDENSELAYER_CUH


#include "../../pch.cuh"
#include "NNLayer.cuh"

//#define NNDebug
using namespace std;
#define l_rate 0.2

class NNDenseLayer : public NNLayer {
public:
    explicit NNDenseLayer(uint32_t _size, mt19937 &_gen, bool _is_random = true, float learning_rate = l_rate) :
            NNLayer(_size, _gen, _is_random, learning_rate) {};

    NNDenseLayer(uint32_t _input_size, uint32_t output_size, mt19937 &_gen, bool _is_random, float learning_rate = l_rate) :
            NNLayer(_input_size, output_size, _gen, _is_random, learning_rate) {};


    NNDenseLayer(uint32_t _input_size, uint32_t output_size, mt19937 &_gen, bool _is_random,
                 Eigen::MatrixXf &weights, Eigen::VectorXf &biases, float learning_rate = l_rate) :
            NNLayer(_input_size, output_size, _gen, _is_random, weights, biases, learning_rate) {};

    void allocate_layer(float = -1.0f, float = 1.0f) override;


    shared_ptr<Eigen::VectorXf> propagate(const shared_ptr<Eigen::VectorXf> &pre_act) override;

    shared_ptr<Eigen::VectorXf> back_propagate(
            const std::shared_ptr<Eigen::VectorXf> &target) override;

    void update_parameters() override;

private:

    constexpr static const auto sigmoid_derivative = [](float x) {
        return x * (1.0f - x);
    };

    static inline float sigmoid_fn(float x) {
        return (1 / (1 + powf(2.7182818284590452353f, -x)));;
    }

    static inline float relu(float x) {
        return max(0.0f, x);
    }

    constexpr static const auto relu_derivative = [](float x) {
        return (x > 0) ? 1.0f : 0.0f;
    };


    static inline float tanh(float x) {
        return std::tanh(x);
    }

    constexpr static const auto tanh_derivative = [](float x) {
        float tanh = std::tanh(x);
        return 1.0f - (tanh * tanh);
    };

};


#endif //NNCPP_NNDENSELAYER_CUH
