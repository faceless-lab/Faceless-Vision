//
// Created by Andrei Nechaev on 3/13/18.
//

#pragma once

#include <opencv2/core/types.hpp>
#include <ostream>

namespace flv {

    template <class T>
    struct Face {

    private:
        int _left;

        int _top;

        int _right;

        int _bottom;

        cv::Point _lt;

        cv::Point _rb;

        cv::Rect2d _bbox;

    public:
        Face<T>(T left, T top, T right, T bottom) {
            _left = static_cast<int>(left);
            _top = static_cast<int>(top);
            _right = static_cast<int>(right);
            _bottom = static_cast<int>(bottom);

            _lt = cv::Point(_left, _top);
            _rb = cv::Point(_right, _bottom);

            _bbox = cv::Rect2d(_lt, _rb);
        };

        virtual ~Face<T>() = default;

        int get_left() const {
            return _left;
        }

        int get_top() const {
            return _top;;
        }

        int get_right() const {
            return _right;
        }

        int get_bottom() const {
            return _bottom;
        }

        cv::Point get_lt() const {
            return _lt;
        }

        cv::Point get_rb() const {
            return _rb;
        }

        cv::Rect2d get_bbox() const {
            return _bbox;
        }
    };

}