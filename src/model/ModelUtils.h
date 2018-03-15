//
// Created by Andrei Nechaev on 3/4/18.
//

#include <string>
#include <opencv2/core/mat.hpp>

#pragma once

namespace flv {

    class ModelUtils {

    public:

        static void blob_from_image(cv::Mat& src, cv::Mat& dst, cv::Size size);
    };
}