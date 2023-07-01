//
// Created by sid on 30/6/23.
//

#ifndef NNCPP_IMAGEDATA_H
#define NNCPP_IMAGEDATA_H

#include <memory>
#include <vector>
#include <filesystem>
#include "Eigen/Core"
#include "opencv2/opencv.hpp"

using namespace std;
/**
 *  @struct  ImageClassifierData
 *  @brief   This structure encapsulates all data required for an image classification task.
 *
 *  @property   trainingLabels       : A vector holding the labels for each image in the 'trainingImages' vector.
 *
 *  @property   trainingLabelNames   : A vector holding the names of the labels for the 'trainingLabels'. The index of a name in this vector corresponds to the label in 'trainingLabels'.
 *
 *  @property   testLabels           : A vector holding the labels for each image in the 'testImages' vector.
 *
 *  @property   testLabelNames       : A vector holding the names of the labels for the 'testLabels'. The index of a name in this vector corresponds to the label in 'testLabels'.
 *
 *  @property   trainingImages       : A vector holding pointers to the training images. Each pointer is a shared_ptr encapsulating a cv::Mat.
 *
 *  @property   testImages           : A vector holding pointers to the test images. Each pointer is a shared_ptr encapsulating a cv::Mat.
 *
 *  @property   trainingData         : A shared pointer to a vector which contains Eigen::MatrixXf data corresponding to the 'trainingImages'. Contains the image data that is to be used for training the model. Note: This is not a deep copy. Instead, it's a mapping from the 'trainingImages'.
 *
 *  @property   testData             : A shared pointer to a vector which contains Eigen::MatrixXf data corresponding to the 'testImages'. Contains the image data that is to be used for testing the model. Note: This is not a deep copy. Instead, it's a mapping from the 'testImages'.
 */
struct ImageClassifierData {
    vector<uint32_t> trainingLabels;
    vector<string> trainingLabelNames;
    vector<uint32_t> testLabels;
    vector<string> testLabelNames;
    vector<shared_ptr<cv::Mat>> trainingImages;
    vector<shared_ptr<cv::Mat>> testImages;
    shared_ptr<Eigen::MatrixXf> trainingData;
    shared_ptr<Eigen::MatrixXf> testData;
};

/**
 *  @function processDirectory
 *  @brief This function traverses a directory (and its subdirectories) and loads images, labels them and collects label names
 *
 *  @param[in]  dirPath : A reference to the path of the directory to be processed.
 *  @param[out] images  : A reference to the vector where pointers to loaded images will be stored. Each pointer is a shared_ptr encapsulating a cv::Mat loaded via cv::imread. If a loaded image is empty (i.e., couldn't be loaded properly), it will not be added to this vector.
 *  @param[out] labels  : A reference to the vector where labels for each image will be stored. A label for an image is essentially the index of the name of parent directory of the image file in 'discoveredLabels'. If the parent directory's name is not in 'discoveredLabels', it is added and its index is used as the label.
 *  @param[out] discoveredLabels : A reference to vector where unique names of parent directories of the image files will be stored.
 *
 *  For each regular file in the directory which is an image file, the function:
 *      - Loads the image and stores a shared_ptr to it in 'images'
 *      - Checks the parent directory of the file in the 'discoveredLabels'. If it there, the index of the directory in 'discoveredLabels' is used as the label and stored in 'labels'. If it's not there, the new directory is added to 'discoveredLabels' and its new index (of the now larger vector) is used as the label and stored in 'labels'.
 *
 *  If there are any directories in the directory provided in 'dirPath', this function calls itself recursively to process them
 *  > Note : recursive calls are still treated as normal dir, ie. recursive folders are treated as unique label
 */
void processDirectory(const filesystem::path& dirPath,
                      vector<shared_ptr<cv::Mat>>& images,
                      vector<uint32_t> &labels,
                      vector<string>& discoveredLabels,
                      size_t maxData,
                      int cv_flags = cv::IMREAD_GRAYSCALE | cv::IMREAD_UNCHANGED
) {
    size_t totalData = 0;

    for (const auto& file : filesystem::directory_iterator(dirPath)) {
        if (totalData >= maxData) {
            break;
        }

        if (file.is_regular_file()) {
            auto image_ptr = make_shared<cv::Mat>(cv::imread(
                    file.path().string(),
                    cv_flags
            ));

            if (!image_ptr->empty()) {
                totalData += 1;
                images.push_back(image_ptr);

                // get parent directory name
                std::string parentDirectory = file.path().parent_path().filename().string();

                // find the directory in the discovered labels
                auto it = std::find(discoveredLabels.begin(), discoveredLabels.end(), parentDirectory);

                if (it != discoveredLabels.end()) {
                    // if the directory was found, subtract begin iterator from it to get the index
                    labels.push_back(std::distance(discoveredLabels.begin(), it));
                }
                else {
                    // if the directory wasn't found, add it to the discovered labels and
                    // use the new size of the vector - 1 as the label
                    discoveredLabels.push_back(parentDirectory);
                    labels.push_back(discoveredLabels.size() - 1);
                }
            }
        } else if(file.is_directory()){
            processDirectory(file.path(), images, labels, discoveredLabels, maxData/10, cv_flags);
        }
    }
}

/**
 *  @function   load_image_data_dir
 *  @brief      This function loads image data from a directory for training and testing purposes.
 *
 *  @param[in]  directoryPath: The path to the directory that contains the training and testing subdirectories.
 *
 *  @return     An instance of the ImageClassifierData struct which contains the image data for training and testing,
 *              corresponding labels, label names, and mapped Eigen::MatrixXf data.
 */
ImageClassifierData load_image_data_dir(const string& directoryPath) {
    string trainingDir = directoryPath + "/training";
    string testingDir = directoryPath + "/testing";

    ImageClassifierData imageData;

    // Load training data
    vector<shared_ptr<cv::Mat>> trainingImages;
    vector<uint32_t> trainingLabels;
    vector<string> discoveredLabels;

    processDirectory(trainingDir, trainingImages, trainingLabels, discoveredLabels, 100);
    imageData.trainingImages = trainingImages;
    imageData.trainingLabels = trainingLabels;
    imageData.trainingLabelNames = discoveredLabels;

    // Load test data
    vector<shared_ptr<cv::Mat>> testImages;
    vector<uint32_t > testLabels;
    discoveredLabels.clear();

    processDirectory(testingDir, testImages, testLabels, discoveredLabels, 100);
    imageData.testImages = testImages;
    imageData.testLabels = testLabels;
    imageData.testLabelNames = discoveredLabels;

    // Convert training data to Eigen::MatrixXf
    auto const image_rows = trainingImages.at(0)->rows;
    auto const image_cols = trainingImages.at(0)->cols;
    auto const area = image_rows * image_cols;

    shared_ptr<Eigen::MatrixXf> trainingData = make_shared<Eigen::MatrixXf>(trainingImages.size(), area);
    auto index = 0;
    for (auto& image : trainingImages) {
        // Map uchar data to ArrayXd
        Eigen::Map<Eigen::Array<unsigned char, Eigen::Dynamic, 1>> array(image->data, area);

        // Cast ArrayXd to VectorXf
        Eigen::VectorXf vector = array.cast<float>();

        // Assign the casted vector to a row in trainingData
        trainingData->row(index) = vector;

        index++;
    }

    // Convert test data to Eigen::MatrixXf
    shared_ptr<Eigen::MatrixXf> testData = make_shared<Eigen::MatrixXf>(testImages.size(), area);
    index = 0;
    for (auto& image : testImages) {
        // Map uchar data to ArrayXd
        Eigen::Map<Eigen::Array<unsigned char, Eigen::Dynamic, 1>> array(image->data, area);

        // Cast ArrayXd to VectorXf
        Eigen::VectorXf vector = array.cast<float>();

        // Assign the cast vector to a row in testData
        testData->row(index) = vector;

        index++;
    }

    // Create and return the struct with shared pointers to vectors of Eigen::MatrixXf
    imageData.trainingData = trainingData;
    imageData.testData = testData;
    return imageData;
}



#endif //NNCPP_IMAGEDATA_H
