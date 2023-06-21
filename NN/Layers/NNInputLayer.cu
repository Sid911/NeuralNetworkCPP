//
// Created by sid on 20/6/23.
//

#include <iostream>
#include "NNInputLayer.cuh"

shared_ptr<Eigen::VectorXf> NNInputLayer::propagate(const shared_ptr<Eigen::VectorXf> &inp) {
    activations = inp;
    cout << "Input Layer vec : " << *activations << "\n";
    return activations;
}

shared_ptr<Eigen::VectorXf> NNInputLayer::back_propagate(const shared_ptr<Eigen::VectorXf> &pre_delta,
                                                         const Eigen::MatrixXf &pre_w) {

    return activations;
}

void NNInputLayer::allocate_layer(float, float) {

}
