//
// Created by sid on 20/6/23.
//

#ifndef NEURALNETWORKCPP_NNLAYER_CUH
#define NEURALNETWORKCPP_NNLAYER_CUH

#include <memory>
#include <cstdint>
#include <random>
#include "Eigen/Core"

using namespace std;

class NNLayer {
public:
    uint32_t input_size, output_size;

    float learning_rate = 0.1f;
    bool is_random;
    bool predefined_WnB = false;

    Eigen::MatrixXf weights;
    Eigen::VectorXf biases
    ;
    shared_ptr<Eigen::VectorXf> delta;
    shared_ptr<Eigen::VectorXf> z_vec;
    shared_ptr<Eigen::VectorXf> activations;

    NNLayer(uint32_t _size, mt19937 &_gen, bool _is_random = true) : input_size(_size), output_size(_size),
                                                                     is_random(_is_random), gen(_gen) {
        z_vec = make_shared<Eigen::VectorXf>();
        activations = make_shared<Eigen::VectorXf>();
    };

    NNLayer(uint32_t _input_size, uint32_t output_size,
            mt19937 &_gen, bool _is_random) :
            input_size(_input_size),
            output_size(output_size),
            is_random(_is_random),
            gen(_gen) {
        z_vec = make_shared<Eigen::VectorXf>();
        activations = make_shared<Eigen::VectorXf>();
    }


    NNLayer(uint32_t _input_size, uint32_t output_size,
            mt19937 &_gen, bool _is_random, Eigen::MatrixXf& weights, Eigen::VectorXf &biases) :
            input_size(_input_size),
            output_size(output_size),
            is_random(_is_random),
            gen(_gen),
            weights(weights),
            biases(biases),
            predefined_WnB(true) {
        z_vec = make_shared<Eigen::VectorXf>();
        activations = make_shared<Eigen::VectorXf>();
    }

    virtual void allocate_layer(float, float) = 0;

    virtual shared_ptr<Eigen::VectorXf> propagate(const shared_ptr<Eigen::VectorXf> &inp) = 0;

    virtual shared_ptr<Eigen::VectorXf> back_propagate(
            const std::shared_ptr<Eigen::VectorXf> &pre_delta) = 0;

    virtual void  update_parameters() = 0;

protected:
    std::mt19937 &gen;
};


#endif //NEURALNETWORKCPP_NNLAYER_CUH
