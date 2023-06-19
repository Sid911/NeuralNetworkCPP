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

#define assertm(exp, msg) assert(((void)msg, exp))

using namespace std;

class NNSequentialModel {
public:
    // Member variables
    vector<NNDenseLayer> layers;
    bool verbose_logs = false;

    explicit NNSequentialModel(vector<NNDenseLayer> _l);

    // Member functions
    Eigen::VectorXf predict(Eigen::VectorXf &inp);

    void train(const vector<shared_ptr<Eigen::VectorXf>> &input,
               const vector<shared_ptr<Eigen::VectorXf>>& labels,
               uint32_t steps);

private:
    void allocate_layers();

};

#endif //NNCPP_NNSEQUENTIALMODEL_CUH
