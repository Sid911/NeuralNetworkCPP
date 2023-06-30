//
// Created by sid on 20/6/23.
//

#include "NNInputLayer.cuh"

shared_ptr<Eigen::VectorXf> NNInputLayer::propagate(const shared_ptr<Eigen::VectorXf> &inp) {
    activations = inp;
    logger << "Input Layer vec : " << *activations << "\n";
    return activations;
}

shared_ptr<Eigen::VectorXf> NNInputLayer::back_propagate(const shared_ptr<Eigen::VectorXf> &target) {

    return activations;
}

void NNInputLayer::allocate_layer(float, float) {

}

void NNInputLayer::update_parameters() {

}
