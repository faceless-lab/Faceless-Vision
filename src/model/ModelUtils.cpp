//
// Created by Andrei Nechaev on 3/4/18.
//

#include "ModelUtils.h"
#include "opencv2/dnn.hpp"

void flv::ModelUtils::blob_from_image(cv::Mat& src, cv::Mat &dst, cv::Size size) {
    dst = cv::dnn::blobFromImage(src, 1.0, size, 1.0);
}
