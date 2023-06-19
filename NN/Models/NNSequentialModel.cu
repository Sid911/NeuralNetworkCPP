//
// Created by sid on 17/6/23.
//

#include <iostream>
#include <cassert>
#include "NNSequentialModel.cuh"

NNSequentialModel::NNSequentialModel(vector<NNDenseLayer> _l): layers(std::move(_l)) {

}

Eigen::VectorXf NNSequentialModel::predict(Eigen::VectorXf &inp_vec) {
    return {};
}

void NNSequentialModel::train( shared_ptr<Eigen::VectorXf> input, Eigen::VectorXf &labels, uint32_t steps) {
    allocate_layers();
    // after allocating layers we train with the input data
    cout << "Layer 0" << "\n\t";
    auto res = layers[0].propagate(input);
    cout << "\n";
    for (auto i = 1; i < layers.size(); i++){
        cout << "Layer " << i << "\n\t";
        res = layers[i].propagate(res);
        cout << "\n";
    }
    // res is the last layer's result
    shared_ptr<Eigen::VectorXf> errors(new Eigen::VectorXf(res->size()));
    // calculate errors
    assertm(labels.size() != (*errors).size(), "labels size != model outputsize");



}


void NNSequentialModel::allocate_layers() {
    std::cout << "Allocating Layers : ";
    for (auto i = 0; i < layers.size(); i++) {
        layers[i].allocate_layer(0.0, 10.0);
        cout << ". ";
    }
    cout << "✅\n";
}