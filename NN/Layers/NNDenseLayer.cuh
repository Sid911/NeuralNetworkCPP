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

using namespace std;


class NNDenseLayer {
public:
    uint32_t input_size, output_size;
    bool is_random;

    explicit NNDenseLayer(uint32_t _size, bool _is_random = true);

    NNDenseLayer(uint32_t _input_size, uint32_t output_size, bool _is_random);

    void allocate_layer(float = 0.0f, float = 1.0f );

    shared_ptr<vector<float>> propagate(shared_ptr<vector<float>> inp);

    shared_ptr<vector<float>> back_propagate(std::shared_ptr<vector<float>> z_vec);

private:
    /*
     * The weights of r rows (each row is for 1 neuron in the layer)
     * The column c represents weights per input.
     * matrix being input_size * output_size
     */
    vector<vector<float>> weights;
    vector<float> biases;
    shared_ptr<vector<float>> values;

    shared_ptr<vector<float>> compute_error(shared_ptr<vector<float>> labels){
        shared_ptr<vector<float>> errors(new vector<float>(output_size));

        assertm(labels->size() == errors->size(), "labels size != model outputsize");
        for (auto i = 0; i < labels->size(); i++){
            auto cost = pow(values->at(i) - labels->at(i), 2);
            errors->push_back(cost);
        }
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
