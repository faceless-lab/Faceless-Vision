//
// Created by Andrei Nechaev on 3/25/18.
//

#include "Detector.h"

flv::detect::Detector::Detector(const std::string &name) : _tracker_name(name), _trackers{} {}

flv::detect::Detector::~Detector() {
    cleanup();
}

void flv::detect::Detector::add(std::string &name, const cv::Ptr<cv::Tracker> &tracker) {
}

void flv::detect::Detector::remove(const std::string &name) {

}

bool flv::detect::Detector::update(cv::InputArray src) {
    if (_trackers.empty()) {
        return false;
    }


    for (auto pair : _trackers) {
        auto tracker_ptr = pair.second;
        if (pair.first >= _bboxes.size() || !tracker_ptr->update(src, _bboxes.at(pair.first))) {
            return false;
        }
    }

    return true;
}

void flv::detect::Detector::track(cv::Mat &frame, std::vector<cv::Rect2d> bboxes) {
    _bboxes = std::vector<cv::Rect2d>(bboxes.begin(), bboxes.end());
    unsigned long i = 0;
    for (const cv::Rect2d &obj : bboxes) {
        auto tracker_ptr = get_tracker();
        auto pair = std::make_pair(i++, tracker_ptr);

        _trackers.insert(pair);
        tracker_ptr->init(frame, obj);
    }
}




