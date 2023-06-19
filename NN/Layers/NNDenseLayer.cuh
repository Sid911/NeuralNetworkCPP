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
    bool is_random;

    explicit NNDenseLayer(uint32_t _size, bool _is_random = true);

    NNDenseLayer(uint32_t _input_size, uint32_t output_size, bool _is_random);

    void allocate_layer(float = 0.0f, float = 1.0f );

    shared_ptr<Eigen::VectorXf > propagate(shared_ptr<Eigen::VectorXf > inp);

    shared_ptr<Eigen::VectorXf > back_propagate(std::shared_ptr<Eigen::VectorXf > z_vec);

private:
    /*
     * The weights of r rows (each row is for 1 neuron in the layer)
     * The column c represents weights per input.
     * matrix being input_size * output_size
     */
    Eigen::MatrixXf weights;
    Eigen::VectorXf  biases;
    shared_ptr<Eigen::VectorXf > values;

    shared_ptr<Eigen::VectorXf > compute_error(shared_ptr<Eigen::VectorXf> labels){
        shared_ptr<Eigen::VectorXf > errors(new Eigen::VectorXf (output_size));

        return errors;
    };

    float sigmoid_fn(float x) {
        return 0.5 * (x / (1 + std::abs(x)) + 1);
    }
    float relu(float x){
        return max(0.0f,x);
    }
};


#endif //NNCPP_NNDENSELAYER_CUH
