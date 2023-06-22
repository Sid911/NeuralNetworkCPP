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
#include "Eigen/Core"
#include "NNLayer.cuh"

using namespace std;


class NNDenseLayer: public NNLayer {
public:
    explicit NNDenseLayer(uint32_t _size, mt19937& _gen, bool _is_random = true):
    NNLayer(_size, _gen, _is_random){};

    NNDenseLayer(uint32_t _input_size, uint32_t output_size, mt19937& _gen, bool _is_random):
            NNLayer(_input_size, output_size, _gen, _is_random){};

    void allocate_layer(float = 0.0f, float = 1.0f ) override;

    shared_ptr<Eigen::VectorXf > propagate(const shared_ptr<Eigen::VectorXf >& pre_act) override;

    shared_ptr<Eigen::VectorXf> back_propagate(
            const std::shared_ptr<Eigen::VectorXf> &target) override ;

    void update_parameters() override;

    constexpr static const auto sigmoid_derivative = [](float x) {
        float sigmoid = 1.0f / (1.0f + std::exp(-x));
        return sigmoid * (1.0f - sigmoid);
    };

private:

    static inline float sigmoid_fn(float x) {
        return 0.5 * (x / (1 + std::abs(x)) + 1);
    }
    static inline float relu(float x){
        return max(0.0f,x);
    }
    constexpr static const auto relu_derivative = [](float x) {
        return (x > 0) ? 1.0f : 0.0f ;};

};



#endif //NNCPP_NNDENSELAYER_CUH
