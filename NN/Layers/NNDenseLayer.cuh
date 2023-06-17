//
// Created by sid on 17/6/23.
//

#ifndef NNCPP_NNDENSELAYER_CUH
#define NNCPP_NNDENSELAYER_CUH

#include <cstdint>
#include <vector>
#include <random>
#include <memory>

using namespace std;

template<typename Tp>
class NNDenseLayer {
public:
    uint32_t input_size, output_size;
    bool is_random;

    explicit NNDenseLayer(uint32_t _size, bool _is_random = true);

    NNDenseLayer(uint32_t _input_size, uint32_t output_size, bool _is_random = true);

    void allocate_layer();

    shared_ptr<vector<Tp>> propagate(shared_ptr<vector<Tp>> inp);

    shared_ptr<vector<Tp>> backpropogate(std::shared_ptr<vector<float>> z_vec);

private:
    vector<vector<float>> weights;
    vector<float> biases;
};


#endif //NNCPP_NNDENSELAYER_CUH
