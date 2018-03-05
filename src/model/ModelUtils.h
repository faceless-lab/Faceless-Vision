//
// Created by Andrei Nechaev on 3/4/18.
//

#pragma once

#include <string>
#include <opencv2/core/mat.hpp>

#define MODEL_NOT_AVAILABLE 1200
#define CONFIDENCE_LEVEL 0.93

namespace flv {
    const std::string FACE_DNN_PROTO = "models/face/deploy.prototxt.txt";
    const std::string FACE_DNN_MODEL = "models/face/res10_300x300_ssd_iter_140000.caffemodel";

    class ModelUtils {

    public:

        static void blob_from_image(cv::Mat& src, cv::Mat& dst, cv::Size size);
    };
}