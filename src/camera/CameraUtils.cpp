//
// Created by Andrei Nechaev on 3/4/18.
//

#include <opencv2/videoio.hpp>
#include <opencv2/videoio/videoio_c.h>
#include <opencv2/imgproc.hpp>
#include "CameraUtils.h"

void CameraUtils::set_up(cv::VideoCapture &cap, cv::Size size) {
    cap.set(CV_CAP_PROP_FRAME_WIDTH, size.width);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, size.height);
}
