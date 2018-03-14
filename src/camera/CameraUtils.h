//
// Created by Andrei Nechaev on 3/4/18.
//

#include <opencv/highgui.h>

#pragma once

#define CAMERA_NOT_AVAILABLE 1100

namespace flv {
    class CameraUtils {

    public:
        static void set_up(cv::VideoCapture& cap, cv::Size size);


    };
}

