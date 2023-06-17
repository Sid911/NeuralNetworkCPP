//
// Created by sid on 17/6/23.
//

#include "NNSequentialModel.cuh"

template<typename Tp>
NNSequentialModel<Tp>::NNSequentialModel(vector<NNDenseLayer<Tp>> _l): layers(std::move(_l)) {

}

template<typename Tp>
vector<float> NNSequentialModel<Tp>::predict(vector<Tp> &inp_vec) {
    return {};
}

template<typename Tp>
void NNSequentialModel<Tp>::train(vector<vector<Tp>> &input, uint32_t steps) {
    allocate_layers();

};


template<typename Tp>
void NNSequentialModel<Tp>::allocate_layers() {
    for (auto &layer: layers) {
        layer.allocate_layer();
    }
}