//
// Created by sid on 17/6/23.
//
#include <iostream>
#include <cassert>
#include <utility>
#include "NNSequentialModel.cuh"

NNSequentialModel::NNSequentialModel(vector<shared_ptr<NNLayer>> _l) : layers(std::move(_l)) {

}

[[maybe_unused]] shared_ptr<Eigen::MatrixXf> NNSequentialModel::predict(Eigen::MatrixXf &inp_vec) {
    shared_ptr<Eigen::MatrixXf> results = make_shared<Eigen::MatrixXf>(inp_vec.rows(), layers.back()->output_size);
    for (auto i = 0; i < inp_vec.rows(); i++){
        auto res = forward(inp_vec.row(i));
        results->row(i) = *res;
    }
    return results;
}

void NNSequentialModel::train(const Eigen::MatrixXf &input,
                              const Eigen::MatrixXf &labels,
                              uint32_t steps) {
    allocate_layers();

    // Log: Allocating layers
    if (verbose_log)
        cout << "--------------------------------------------------------\n"
             << "Allocating layers\n"
             << "--------------------------------------------------------\n";

    for (uint32_t step = 0; step < steps; ++step) {
        cout << "Training Step: " << (step + 1) << "/" << steps << "\n";

        double total_loss = 0.0;

        for (uint32_t inp_index = 0; inp_index < input.rows(); inp_index++) {
            // Log: Training step
            if (verbose_log) {
                cout << "Input :" << input.row(inp_index) << "\t Expected out : " << labels.row(inp_index);
                cout << "\n";
            }
            shared_ptr<Eigen::Matrix<float, -1, 1>> res = forward(input.row(inp_index));
            total_loss += (*res - labels.row(inp_index)).squaredNorm();
            back(labels.row(inp_index), res);
        }

        cout << "Average loss MSE : " << total_loss / labels.size() << "\n";
    }
}

void NNSequentialModel::back(const Eigen::VectorXf &label,
                             const shared_ptr<Eigen::Matrix<float, -1, 1>> &res) {
    // res is the last layer's result
    // Log: Backpropagation start
    if (verbose_log)
        cout << "--------------------------------------------------------\n"
             << "\tBackpropagation start\n";

    // calculate first error and delta
    shared_ptr<Eigen::VectorXf> error = make_shared<Eigen::VectorXf>(*res - label);

    if (verbose_log)cout << "Layer " << layers.size() - 1 << " Errors : \n";

    auto next_labels = layers.back()->back_propagate(error);

    for (long long i = layers.size() - 2; i > 0; i--) {
        if (verbose_log)cout << "Layer " << i << " Errors : \n";
        next_labels = layers.at(i)->back_propagate(next_labels);
        if (verbose_log)cout << "\n\n";
    }

    for (auto i = 1; i < layers.size(); i++) {
        layers[i]->update_parameters();
    }
    // Log: Backpropagation complete
    if (verbose_log)
        cout << "\tBackpropagation complete\n"
             << "--------------------------------------------------------\n";
}

shared_ptr<Eigen::Matrix<float, -1, 1>>
NNSequentialModel::forward(const Eigen::VectorXf &input) {// Forward propagation
    if (verbose_log)
        cout << "--------------------------------------------------------\n"
             << "\tPropagation start\n";
    if (verbose_log)cout << "Layer 0 \n";
    auto prev_act = make_shared<Eigen::VectorXf>(input);
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
        layer->allocate_layer(-1.0, 1.0);
        cout << ". ";
    }
    cout << "✅\n" << "Layer Weights : \n";
    for (auto &layer: layers) {
        cout << "\t" << layer->weights.rows() << " x " << layer->weights.cols() << "\n";
        cout << layer-> weights << "\n";
    }
}