//
// Created by Andrei Nechaev on 3/19/18.
//

#pragma once

#include <opencv2/core/types.hpp>
#include <opencv2/core/mat.hpp>

bool is_inside(cv::Rect2d &rect, cv::Mat &mat) {
    return (rect & cv::Rect2d(0, 0, mat.cols, mat.rows)) == rect;
}
