//
// Created by sid on 17/6/23.
//

#include <iostream>
#include "NNSequentialModel.cuh"

template<typename Tp>
NNSequentialModel<Tp>::NNSequentialModel(vector<NNDenseLayer<Tp>> _l): layers(std::move(_l)) {

}

template<typename Tp>
vector<float> NNSequentialModel<Tp>::predict(vector<Tp> &inp_vec) {
    return {};
}

template<typename Tp>
void NNSequentialModel<Tp>::train( vector<Tp> &input, vector<Tp> &labels, uint32_t steps) {
    allocate_layers();
    // after allocating layers we train with the input data
    auto res = layers[0].propagate(input);
    for (auto i = 1; i < layers.size(); i++){
        res = layers[i].propagate(res);
    }
    // res is the last layer's result
};


template<typename Tp>
void NNSequentialModel<Tp>::allocate_layers() {
    std::cout << "Allocating Layers : ";
    for (const NNDenseLayer<Tp> &layer: layers) {
        layer.allocate_layer();
        cout << ". ";
    }
    cout << "✅";
}