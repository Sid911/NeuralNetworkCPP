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

    NNDenseLayer(uint32_t _size, bool _is_random = true);

    NNDenseLayer(uint32_t _input_size, uint32_t output_size, bool _is_random);

    void allocate_layer();

    const vector<Tp>& propagate(const vector<Tp> &inp);

    shared_ptr<vector<Tp>> back_propagate(std::shared_ptr<vector<float>> z_vec);

private:
    /*
     * The weights of r rows (each row is for 1 neuron in the layer)
     * The column c represents weights per input.
     * matrix being input_size * output_size
     */
    vector<vector<float>> weights;
    vector<float> biases;
    shared_ptr<vector<float>> values;
    float sigmoid_fn(int x) {
        return 0.5 * (x / (1 + std::abs(x)) + 1);
    }
};




#endif //NNCPP_NNDENSELAYER_CUH
