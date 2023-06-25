//
// Created by sid on 17/6/23.
//
#pragma once
#ifndef NNCPP_NNSEQUENTIALMODEL_CUH
#define NNCPP_NNSEQUENTIALMODEL_CUH
#define EIGEN_NO_CUDA

#include <utility>
#include <vector>
#include <cstdint>
#include "../Layers/NNDenseLayer.cuh"

#define assertm(exp, msg) assert(((void)msg, exp))

using namespace std;

class NNSequentialModel {
public:
    // Member variables
    vector<shared_ptr<NNLayer>> layers;
    bool verbose_log = false;

    explicit NNSequentialModel(vector<shared_ptr<NNLayer>> _l);

    // Member functions
    [[maybe_unused]] shared_ptr<Eigen::MatrixXf> predict(Eigen::MatrixXf &inp);

    void train(const Eigen::MatrixXf &input,
               const Eigen::MatrixXf &labels,
               uint32_t steps);

private:
    void allocate_layers();

    shared_ptr<Eigen::Matrix<float, -1, 1>>
    forward(const Eigen::VectorXf &input);

    void back(const Eigen::VectorXf &label,
              const shared_ptr<Eigen::Matrix<float, -1, 1>> &res);
};

#endif //NNCPP_NNSEQUENTIALMODEL_CUH
