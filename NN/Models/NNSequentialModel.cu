//
// Created by sid on 17/6/23.
//
#include <iostream>
#include <cassert>
#include <utility>
#include "NNSequentialModel.cuh"

#define ANSI_S "\033[1;"
#define ANSI_R "\033[0m"

/*
F   B   Color
30	40	Black
31	41	Red
32	42	Green
33	43	Yellow
34	44	Blue
35	45	Magenta
36	46	Cyan
37	47	White
90	100	Bright Black (Gray)
91	101	Bright Red
92	102	Bright Green
93	103	Bright Yellow
94	104	Bright Blue
95	105	Bright Magenta
96	106	Bright Cyan
97	107	Bright White
 */
NNSequentialModel::NNSequentialModel(vector<shared_ptr<NNLayer>> _l) : layers(std::move(_l)) {

}

[[maybe_unused]] shared_ptr<Eigen::MatrixXf> NNSequentialModel::predict(Eigen::MatrixXf &inp_vec) {
    shared_ptr<Eigen::MatrixXf> results = make_shared<Eigen::MatrixXf>(inp_vec.rows(), layers.back()->output_size);
    cout << "\n\033[1;30m\033[1;107m ----- Predications ----- \033[0m\n\n";
    for (auto i = 0; i < inp_vec.rows(); i++) {
        auto res = forward(inp_vec.row(i));
        results->row(i) = *res;

        std::cout << "Prediction :\033[1;100m " << inp_vec.row(i) << " \033[0m Out :\033[1;40m "
                  << *res << " \033[0m\n";
    }
    return results;
}

void NNSequentialModel::train(const Eigen::MatrixXf &input,
                              const Eigen::MatrixXf &labels,
                              uint32_t steps) {
    allocate_layers();

    // Log: Allocating layers
#ifdef NDebug
    cout << "--------------------------------------------------------\n"
         << "Allocating layers\n"
         << "--------------------------------------------------------\n";
#endif
    for (uint32_t step = 0; step < steps; ++step) {
        cout << "\033[1;90mTraining Step: " << (step + 1) << "/" << steps << "\033[0m\n";

        double total_loss = 0.0;

        for (uint32_t inp_index = 0; inp_index < input.rows(); inp_index++) {
            // Log: Training step
#ifdef NDebug
            cout << "Input :" << input.row(inp_index) << "\t Expected out : " << labels.row(inp_index);
            cout << "\n";
#endif
            shared_ptr<Eigen::Matrix<float, -1, 1>> res = forward(input.row(inp_index));
            total_loss += (*res - labels.row(inp_index)).squaredNorm();
            back(labels.row(inp_index), res);
        }

        cout << "Average loss MSE :\033[1;40m " << total_loss / labels.size() << " \033[0m\n";
    }
}

void NNSequentialModel::back(const Eigen::VectorXf &label,
                             const shared_ptr<Eigen::Matrix<float, -1, 1>> &res) {
    // res is the last layer's result
    // Log: Backpropagation start

#ifdef NDebug
    cout << "--------------------------------------------------------\n"
         << "\tBackpropagation start\n";
#endif
    // calculate first error and delta
    shared_ptr<Eigen::VectorXf> error = make_shared<Eigen::VectorXf>(*res - label);

#ifdef NDebug
    cout << "Layer " << layers.size() - 1 << " Errors : \n";
#endif
    auto next_labels = layers.back()->back_propagate(error);

    for (long long i = layers.size() - 2; i > 0; i--) {
#ifdef NDebug
        cout << "Layer " << i << " Errors : \n";
#endif
        next_labels = layers.at(i)->back_propagate(next_labels);

#ifdef NDebug
        cout << "\n\n";
#endif
    }

    for (auto i = 1; i < layers.size(); i++) {
        layers[i]->update_parameters();
    }
    // Log: Backpropagation complete
#ifdef NDebug
    cout << "\tBackpropagation complete\n"
         << "--------------------------------------------------------\n";
#endif
}

shared_ptr<Eigen::VectorXf>
NNSequentialModel::forward(const Eigen::VectorXf &input) {// Forward propagation
#ifdef NDebug
    cout << "--------------------------------------------------------\n"
         << "\tPropagation start\n";
cout << "Layer 0 \n";
#endif
    auto prev_act = make_shared<Eigen::VectorXf>(input);
    for (auto i = 0; i < layers.size(); i++) {
#ifdef NDebug
        cout << "Layer " << i << "\n";
#endif
        prev_act = layers[i]->propagate(prev_act);
#ifdef NDebug
        cout << "\n";
#endif
    }

#ifdef NDebug
    cout << "\tPropagation complete\n"
         << "--------------------------------------------------------\n";
#endif
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
        cout << layer->weights << "\n";
    }
}