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
#include "../Utils/Logger.cuh"

using namespace std;

class NNSequentialModel {
public:
    // Member variables
    vector<shared_ptr<NNLayer>> layers;
    NN::Logger logger;

    explicit NNSequentialModel(vector<shared_ptr<NNLayer>> _l);

    // Member functions
    [[maybe_unused]] shared_ptr<Eigen::MatrixXf> predict(Eigen::MatrixXf &inp);

    void test(const Eigen::MatrixXf &input,
              const Eigen::MatrixXf &output);

    void test_classification(const Eigen::MatrixXf &input,
              const Eigen::MatrixXf &labels,
              const vector<string> &labelNames);

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
