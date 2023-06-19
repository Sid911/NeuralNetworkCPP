//
// Created by sid on 17/6/23.
//

#include <iostream>
#include <cassert>
#include "NNSequentialModel.cuh"

NNSequentialModel::NNSequentialModel(vector<NNDenseLayer> _l) : layers(std::move(_l)) {

}

Eigen::VectorXf NNSequentialModel::predict(Eigen::VectorXf &inp_vec) {
    return {};
}

void NNSequentialModel::train(const vector<shared_ptr<Eigen::VectorXf>> &input,
                              const vector<shared_ptr<Eigen::VectorXf>> &labels,
                              uint32_t steps) {
    allocate_layers();

    // Log: Allocating layers
    if (verbose_logs)
        cout << "--------------------------------------------------------\n"
             << "Allocating layers\n"
             << "--------------------------------------------------------\n";

    for (uint32_t step = 0; step < steps; ++step) {
        // Log: Training step
        cout << "--------------------------------------------------------\n"
             << "Training Step: " << (step + 1) << "/" << steps << "\n"
             << "--------------------------------------------------------\n";

        // Forward propagation
        if (verbose_logs)
            cout << "--------------------------------------------------------\n"
                 << "\tPropagation start\n"
                 << "--------------------------------------------------------\n";

        shared_ptr<Eigen::Matrix<float, -1, 1>> res = layers[0].propagate(input.at(0));
        cout << "\n";
        for (auto i = 1; i < layers.size(); i++) {
            if (verbose_logs)cout << "Layer " << i << "\n";
            res = layers[i].propagate(res);
            if (verbose_logs)cout << "\n";
        }
        if (verbose_logs)
            cout << "--------------------------------------------------------\n"
                 << "\tPropagation complete\n"
                 << "--------------------------------------------------------\n";

        // res is the last layer's result
        assertm(labels.size() != res->size(), "labels size != model output size");

        // Log: Backpropagation start
        if (verbose_logs)
            cout << "--------------------------------------------------------\n"
                 << "\tBackpropagation start\n"
                 << "--------------------------------------------------------\n";

        cout << "Current Error: ";
        auto next_labels = layers.back().back_propagate(labels.back());
        for (long long i = layers.size() - 2; i >= 0; i--) {
            cout << "Layer " << i << " Errors : \n";
            next_labels = layers.at(i).back_propagate(next_labels);
            cout << "\n";
        }

        // Log: Backpropagation complete
        if (verbose_logs)
            cout << "--------------------------------------------------------\n"
                 << "\tBackpropagation complete\n"
                 << "--------------------------------------------------------\n";
    }
}


void NNSequentialModel::allocate_layers() {
    std::cout << "Allocating Layers : ";
    for (auto &layer: layers) {
        layer.allocate_layer(0.0, 10.0);
        cout << ". ";
    }
    cout << "✅\n" << "Layer Weights : \n";
    for (auto &layer: layers) {
        cout << "\t" << layer.weights.rows() << " x " << layer.weights.cols() << "\n\n";
    }
}