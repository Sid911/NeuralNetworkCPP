//
// Created by sid on 17/6/23.
//

#ifndef NNCPP_NNSEQUENTIALMODEL_CUH
#define NNCPP_NNSEQUENTIALMODEL_CUH

#include <utility>
#include <vector>
#include <cstdint>
#include "../Layers/NNDenseLayer.cuh"

using namespace std;

template<typename Tp>
class NNSequentialModel {
public:
    // Member variables
    const vector<NNDenseLayer<Tp>> layers;

    explicit NNSequentialModel(vector<NNDenseLayer<Tp>> _l);

    // Member functions
    vector<float> predict(vector<Tp> &inp);

    void train(vector<Tp> &input,vector<Tp> &labels, uint32_t steps);

private:
    void allocate_layers();

};


enum NNModelError {
    AllocationError,

};
#endif //NNCPP_NNSEQUENTIALMODEL_CUH
