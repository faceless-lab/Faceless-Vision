//
// Created by Andrei Nechaev on 3/25/18.
//

#pragma once

#include <map>
#include <opencv2/tracking.hpp>

namespace flv {

    namespace detect {
        class Detector {
        private:
            std::string _tracker_name;
            std::map<unsigned long, cv::Ptr<cv::Tracker>> _trackers;
            std::vector<cv::Rect2d> _bboxes;

            inline cv::Ptr<cv::Tracker> get_tracker() {
                if (_tracker_name == "MIL") {
                    return cv::TrackerMIL::create();
                }

                return cv::TrackerKCF::create();
            }

        public:
            explicit Detector(const std::string &name);

            virtual ~Detector();

            inline void cleanup() {
                for (auto pair : _trackers) {
                    pair.second->empty();
                    pair.second.release();
                }
                _bboxes.clear();
            }

            void track(cv::Mat &frame, std::vector<cv::Rect2d> bboxes);

            void add(std::string& name, const cv::Ptr<cv::Tracker> &tracker);

            void remove(const std::string& name);

            bool update(cv::InputArray src);

            std::vector<cv::Rect2d> get_bboxes() const { return _bboxes; }
        };
    }
}
