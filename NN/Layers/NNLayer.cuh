//
// Created by sid on 20/6/23.
//

#ifndef NEURALNETWORKCPP_NNLAYER_CUH
#define NEURALNETWORKCPP_NNLAYER_CUH

#include "../../pch.cuh"
#include "../Utils/Logger.cuh"

using namespace std;

/**
 *  @class NNLayer
 *  @brief This class represents a layer in a neural network.
 *
 *  Properties:
 *  @property   input_size   : The size of the input to this layer.
 *  @property   output_size  : The size of the output from this layer.
 *  @property   learning_rate: The learning rate for this layer.
 *  @property   is_random    : A flag to indicate whether initialization is random or not.
 *  @property   predefined_WnB: A flag to indicate whether weights and biases are predefined.
 *  @property   weights      : The weights in this layer.
 *  @property   biases       : The biases in this layer.
 *  @property   delta        : Shared pointer to delta values.
 *  @property   z_vec        : Shared pointer to z vector.
 *  @property   activations  : Shared pointer to activations.
 *
 *
 *  Constructors:
 *  @constructor NNLayer(_size, _gen, _is_random): Initializes an NNLayer with equal input and output sizes.
 *  @constructor NNLayer(_input_size, output_size, _gen, _is_random): Initializes an NNLayer with specified input, output sizes.
 *  @constructor NNLayer(_input_size, output_size, _gen, _is_random, weights, biases): Initializes an NNLayer with specified input, output sizes and pre-defined weights and biases.
 *
 *  Functions:
 *  @function   allocate_layer: Virtual function to allocate parameters for this layer.
 *  @function   propagate: Virtual function to propagate the input through this layer.
 *  @function   back_propagate: Virtual function to perform backpropagation through this layer.
 *  @function   update_parameters: Virtual function to update the parameters of this layer.
 */
class NNLayer {
public:
    uint32_t input_size, output_size;

    float learning_rate = 0.2;
    bool is_random;
    bool predefined_WnB = false;

    Eigen::MatrixXf weights;
    Eigen::VectorXf biases;

    shared_ptr<Eigen::VectorXf> delta;
    shared_ptr<Eigen::VectorXf> z_vec;
    shared_ptr<Eigen::VectorXf> activations;

    NN::Logger logger;

    /**
     *
     * @param _size [in] : Sets output_size equal to input size
     * @param _gen [in] : mt19937 generator required for random generator
     * @param _is_random [in] : sets initialization as random or zero
     *
     *
     */
    NNLayer(uint32_t _size, mt19937 &_gen, bool _is_random = true) : input_size(_size), output_size(_size),
                                                                     is_random(_is_random), gen(_gen),
                                                                     logger(false){
        z_vec = make_shared<Eigen::VectorXf>();
        activations = make_shared<Eigen::VectorXf>();
    };

    NNLayer(uint32_t _input_size, uint32_t output_size,
            mt19937 &_gen, bool _is_random) :
            input_size(_input_size),
            output_size(output_size),
            is_random(_is_random),
            gen(_gen), logger(false) {
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
            predefined_WnB(true),
            logger(false){
        z_vec = make_shared<Eigen::VectorXf>();
        activations = make_shared<Eigen::VectorXf>();
    }

    virtual void allocate_layer(float, float) = 0;

    virtual shared_ptr<Eigen::VectorXf> propagate(const shared_ptr<Eigen::VectorXf> &inp) = 0;

    virtual shared_ptr<Eigen::VectorXf> back_propagate(
            const std::shared_ptr<Eigen::VectorXf> &pre_delta) = 0;

    virtual void  update_parameters() = 0;

    virtual ~NNLayer() {};

protected:
    std::mt19937 &gen;
};


#endif //NEURALNETWORKCPP_NNLAYER_CUH
