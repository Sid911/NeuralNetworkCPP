//
// Created by sid on 17/6/23.
//
#pragma once
#ifndef NNCPP_NNSEQUENTIALMODEL_CUH
#define NNCPP_NNSEQUENTIALMODEL_CUH

#include <utility>
#include <vector>
#include <cstdint>
#include "../Layers/NNDenseLayer.cuh"

using namespace std;

class NNSequentialModel {
public:
    // Member variables
    vector<NNDenseLayer> layers;

    explicit NNSequentialModel(vector<NNDenseLayer> _l);

    // Member functions
    vector<float> predict(vector<float> &inp);

    void train(vector<float> &input,vector<float> &labels, uint32_t steps);

private:
    void allocate_layers();

};

enum NNModelError {
    AllocationError,

};
#endif //NNCPP_NNSEQUENTIALMODEL_CUH
