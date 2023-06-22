//
// Created by sid on 17/6/23.
//

#include <iostream>
#include <cassert>
#include <utility>
#include "NNSequentialModel.cuh"

NNSequentialModel::NNSequentialModel(vector<shared_ptr<NNLayer>> _l) : layers(std::move(_l)) {

}

[[maybe_unused]] shared_ptr<Eigen::VectorXf> NNSequentialModel::predict(shared_ptr<Eigen::VectorXf> &inp_vec) {
    return forward(inp_vec);
}

void NNSequentialModel::train(const vector<shared_ptr<Eigen::VectorXf>> &input,
                              const vector<shared_ptr<Eigen::VectorXf>> &labels,
                              uint32_t steps) {
    allocate_layers();

    // Log: Allocating layers
    if (verbose_log)
        cout << "--------------------------------------------------------\n"
             << "Allocating layers\n"
             << "--------------------------------------------------------\n";

    for (uint32_t step = 0; step < steps; ++step) {
        cout << "--------------------------------------------------------\n"
             << "Training Step: " << (step + 1) << "/" << steps << "\n"
             << "--------------------------------------------------------\n";

        double total_loss = 0.0;

        for (uint32_t inp_index = 0; inp_index < input.size(); inp_index++) {
            // Log: Training step
            if (verbose_log) {
                cout << "Input :" << *input[inp_index] << "\t Expected out : " << *labels[inp_index];
                cout << "\n";
            }
            shared_ptr<Eigen::Matrix<float, -1, 1>> res = forward(input.at(inp_index));
            total_loss += (*res - *labels[inp_index]).squaredNorm();
            back(labels, inp_index, res);
        }

        cout << "Average loss MSE : " << total_loss/ labels.size() << "\n";
    }
}

void NNSequentialModel::back(const vector<shared_ptr<Eigen::VectorXf>> &labels, uint32_t inp_index,
                             const shared_ptr<Eigen::Matrix<float, -1, 1>> &res) {// res is the last layer's result
    assertm(labels.size() != res->size(), "labels size != model output size");

    // Log: Backpropagation start
    if (verbose_log)
        cout << "--------------------------------------------------------\n"
             << "\tBackpropagation start\n";

    // calculate first error and delta
    shared_ptr<Eigen::VectorXf> error = make_shared<Eigen::VectorXf>(*res - *(labels[inp_index]));

    if (verbose_log)cout << "Layer " << layers.size() - 1 << " Errors : \n";

    auto next_labels = layers.back()->back_propagate(error);

    for (long long i = layers.size() - 2; i > 0; i--) {
        if (verbose_log)cout << "Layer " << i << " Errors : \n";
        next_labels = layers.at(i)->back_propagate(next_labels);
        if (verbose_log)cout << "\n\n";
    }

    for (auto i = 1; i < layers.size(); i++){
        layers[i]->update_parameters();
    }
    // Log: Backpropagation complete
    if (verbose_log)
        cout << "\tBackpropagation complete\n"
             << "--------------------------------------------------------\n";
}

shared_ptr<Eigen::Matrix<float, -1, 1>>
NNSequentialModel::forward(const shared_ptr<Eigen::VectorXf> &input) {// Forward propagation
    if (verbose_log)
        cout << "--------------------------------------------------------\n"
             << "\tPropagation start\n";
    if (verbose_log)cout << "Layer 0 \n";
    auto prev_act = input;
    if (verbose_log)cout << "\n";
    for (auto i = 0; i < layers.size(); i++) {
        if (verbose_log)cout << "Layer " << i << "\n";
        prev_act = layers[i]->propagate(prev_act);
        if (verbose_log)cout << "\n";
    }

    if (verbose_log)
        cout << "\tPropagation complete\n"
             << "--------------------------------------------------------\n";
    return prev_act;
}


void NNSequentialModel::allocate_layers() {
    std::cout << "Allocating Layers : ";
    for (auto &layer: layers) {
        layer->allocate_layer(0.0, 10.0);
        cout << ". ";
    }
    cout << "✅\n" << "Layer Weights : \n";
    for (auto &layer: layers) {
        cout << "\t" << layer->weights.rows() << " x " << layer->weights.cols() << "\n\n";
    }
}