//
// Created by sid on 20/6/23.
//

#ifndef NNCPP_NNINPUTLAYER_CUH
#define NNCPP_NNINPUTLAYER_CUH


#include "NNLayer.cuh"
#define assertm(exp, msg) assert(((void)msg, exp))

class NNInputLayer: public NNLayer{
public:
    NNInputLayer(uint32_t _input_size, uint32_t output_size, mt19937& _gen, bool _is_random):
            NNLayer(_input_size, output_size, _gen, _is_random){};

    shared_ptr<Eigen::VectorXf> propagate(const shared_ptr<Eigen::VectorXf> &inp) override;

    shared_ptr<Eigen::VectorXf>
    back_propagate(const shared_ptr<Eigen::VectorXf> &target) override;

    void allocate_layer(float, float) override;

    void update_parameters() override;
};


#endif //NNCPP_NNINPUTLAYER_CUH
