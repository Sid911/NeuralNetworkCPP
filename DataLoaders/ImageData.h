//
// Created by sid on 30/6/23.
//

#ifndef NNCPP_IMAGEDATA_H
#define NNCPP_IMAGEDATA_H

#include <memory>
#include <vector>
#include <filesystem>
#include "Eigen/Core"
#include "unsupported/Eigen/CXX11/Tensor"
#include "opencv2/opencv.hpp"

using namespace std;

struct ImageData {
    vector<shared_ptr<cv::Mat>> trainingImages;
    vector<shared_ptr<cv::Mat>> testImages;
    shared_ptr<vector<Eigen::MatrixXf>> trainingData;
    shared_ptr<vector<Eigen::MatrixXf>> testData;
};

Eigen::MatrixXf createEigenMatrix(const string &filename) {
    cv::Mat image = cv::imread(filename, cv::IMREAD_UNCHANGED);
    Eigen::MatrixXf matrix;
    matrix = Eigen::Map<Eigen::MatrixXf>(image.ptr<float>(), image.rows, image.cols);
    return matrix;
}


ImageData load_image_data_dir(const string& directoryPath) {
    string trainingDir = directoryPath + "/training";
    string testingDir = directoryPath + "/testing";

    ImageData imageData;

    // Load training data
    vector<shared_ptr<cv::Mat>> trainingImages;
    for (const auto& file : filesystem::directory_iterator(trainingDir)) {
        if (file.is_regular_file()) {
            auto image_ptr = make_shared<cv::Mat>(cv::imread(
                    file.path().string(),
                    cv::IMREAD_COLOR | cv::IMREAD_UNCHANGED
            ));
            if (!image_ptr->empty()) {
                trainingImages.push_back(image_ptr);
            }
        }
    }

    // Load test data
    vector<shared_ptr<cv::Mat>> testImages;
    for (const auto& file : filesystem::directory_iterator(testingDir)) {
        if (file.is_regular_file()) {
            auto image_ptr = make_shared<cv::Mat>(cv::imread(
                    file.path().string(),
                    cv::IMREAD_COLOR | cv::IMREAD_UNCHANGED
            ));
            if (!image_ptr->empty()) {
                testImages.push_back(image_ptr);
            }
        }
    }

    // Convert training data to Eigen::MatrixXf
    shared_ptr<vector<Eigen::MatrixXf>> trainingData = make_shared<vector<Eigen::MatrixXf>>();
    trainingData->reserve(trainingImages.size());
    for (auto& image : trainingImages) {
        Eigen::MatrixXf matrix = Eigen::Map<Eigen::MatrixXf>(image->ptr<float>(), image->rows, image->cols * 3);
        trainingData->push_back(matrix);
    }

    // Convert test data to Eigen::MatrixXf
    shared_ptr<vector<Eigen::MatrixXf>> testData = make_shared<vector<Eigen::MatrixXf>>();
    testData->reserve(testImages.size());
    for (auto& image : testImages) {
        Eigen::MatrixXf matrix = Eigen::Map<Eigen::MatrixXf>(image->ptr<float>(), image->rows, image->cols * 3);
        testData->push_back(matrix);
    }

    // Create and return the struct with shared pointers to vectors of Eigen::MatrixXf
    imageData.trainingData = trainingData;
    imageData.testData = testData;
    return imageData;
}



#endif //NNCPP_IMAGEDATA_H
