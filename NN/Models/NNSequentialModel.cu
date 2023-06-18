//
// Created by sid on 17/6/23.
//

#include <iostream>
#include "NNSequentialModel.cuh"

NNSequentialModel::NNSequentialModel(vector<NNDenseLayer> _l): layers(std::move(_l)) {

}

vector<float> NNSequentialModel::predict(vector<float> &inp_vec) {
    return {};
}

void NNSequentialModel::train( vector<float> &input, vector<float> &labels, uint32_t steps) {
    allocate_layers();
    // after allocating layers we train with the input data
    auto res = layers[0].propagate(input);
    for (auto i = 1; i < layers.size(); i++){
        res = layers[i].propagate(res);
    }
    // res is the last layer's result
};


void NNSequentialModel::allocate_layers() {
    std::cout << "Allocating Layers : ";
    for (auto i = 0; i < layers.size(); i++) {
        layers[i].allocate_layer();
        cout << ". ";
    }
    cout << "✅";
}