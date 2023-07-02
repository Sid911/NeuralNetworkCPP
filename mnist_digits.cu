//
// Created by sid on 1/7/23.
//
// Main program
#include "pch.cuh"
#include "NN/Utils/Logger.cuh"
#include "NN/Models/NNSequentialModel.cuh"
#include "NN/Layers/NNInputLayer.cuh"
#include "DataLoaders/ImageData.h"

using namespace Eigen;

int main(){
    NN::Logger logger(true);
    auto data = load_image_data_dir("./data/mnist_png");
    std::cout << data.trainingData->size() << " "<< data.testData->size() << "\n";

    mt19937 gen(time(nullptr));

    vector<shared_ptr<NNLayer>> layers = {
            make_shared<NNInputLayer>(NNInputLayer(784, 784, gen, true)),
            make_shared<NNDenseLayer>(NNDenseLayer(784, 128, gen, true)),
            make_shared<NNDenseLayer>(NNDenseLayer(128, 128, gen, true)),
            make_shared<NNDenseLayer>(NNDenseLayer(128, 10, gen, true)),
    };
    NNSequentialModel model = NNSequentialModel(layers);

    MatrixXf labels(data.trainingLabels.size(),10);
    labels.setZero();

    for(auto i =0; i < data.trainingLabels.size(); i++){
        uint32_t label = stoul(data.trainingLabelNames.at(data.trainingLabels.at(i)));
        if (label > 9 ) return -1;
        labels(i,label ) = 1.0f;
//        logger << data.trainingData->row(i) << "\n" << label << "\n\n";
    }

    model.train(*data.trainingData, labels, 500);
    labels.setZero();

    for(auto i =0; i < data.testLabels.size(); i++){
        uint32_t label = stoul(data.testLabelNames.at(data.testLabels.at(i)));
        if (label > 9 ) return -1;
        labels(i, label ) = 1.0f;
    }
//    model.predict(*data.testData);
    logger << data.testLabelNames;
    model.test_classification(*data.testData, labels, data.trainingLabelNames);
}
