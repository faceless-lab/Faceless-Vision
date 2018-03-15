#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/dnn/shape_utils.hpp>
#include "src/camera/CameraUtils.h"
#include "src/model/ModelUtils.h"
#include "src/types/Face.h"
#include "src/utils/Constants.h"

#define TOP_LEFT_CORNER cv::Point(25, 25)
#define DARK_GREEN cv::Scalar(50, 180, 0)
#define DARK_BLUE cv::Scalar(180, 50, 0)

const int width = 300;
const int height = 300;

int main() {
    auto cap = cv::VideoCapture(0);

    if (!cap.isOpened()) {
        return CAMERA_NOT_AVAILABLE;
    }

    flv::CameraUtils::set_up(cap, cv::Size(width, height));

    auto fps = static_cast<int>(cap.get(CV_CAP_PROP_FPS));
    auto net = cv::dnn::readNetFromCaffe(FACE_DNN_PROTO, FACE_DNN_MODEL);

    if (net.empty()) {
        return MODEL_NOT_AVAILABLE;
    }

    cv::CascadeClassifier eye_csf;
    if (!eye_csf.load(EYE_HAAR_CASCADE)) {
        return EYE_HAAR_CASCADE_NOT_AVAILABLE;
    }

    cv::Ptr<cv::Tracker> tracker = cv::TrackerKCF::create();
    cv::Rect2d bbox;
    bool tracking = false;
    while (true) {
        int64 start = cv::getTickCount();

        cv::Mat frame{};
        if (!cap.read(frame)) {
            std::cerr << "Not able to read frame\n";
            continue;
        }
        cv::flip(frame, frame, 1);

        if (!tracking) {
            cv::Mat blob;
            flv::ModelUtils::blob_from_image(frame, blob, cv::Size(width, height));

            net.setInput(blob);
            auto res = net.forward();

            res = res.reshape(1, static_cast<int>(res.total() / 7));

            for (int i = 0; i < res.rows; i++) {
                float confidence = res.at<float>(i, 2);

                if (confidence > CONFIDENCE_LEVEL) {
                    const auto left = res.at<float>(i, 3) * frame.cols;
                    const auto top = res.at<float>(i, 4) * frame.rows;
                    const auto right = res.at<float>(i, 5) * frame.cols;
                    const auto bottom = res.at<float>(i, 6) * frame.rows;

                    flv::Face<float> face(left, top, right, bottom);

                    const auto lt = face.get_lt();
                    cv::rectangle(frame, lt, face.get_rb(), DARK_GREEN);

                    std::stringstream stream;
                    stream << "Confidence:" << std::setprecision(2) << confidence;
                    cv::putText(frame, stream.str(), lt, CV_FONT_NORMAL, 0.5, DARK_GREEN);

                    bbox = face.get_bbox();
                    tracker->init(frame, bbox);
                }
            }

            blob.release();
            res.release();
        }

        // TODO: update to track multiple faces
        if (tracker->update(frame, bbox)) {
            tracking = true;

            std::vector<cv::Rect> eyes;

            auto face = frame(bbox);
            // TODO: too much noise -> use dlib
            eye_csf.detectMultiScale(face, eyes);

            for (const auto &eye : eyes) {
                cv::Point eye_center(static_cast<int>(bbox.x + eye.x + eye.width / 2),
                                     static_cast<int>(bbox.y + eye.y + eye.height / 2));
                int radius = cvRound((eye.width + eye.height) * 0.25);
                cv::circle(frame, eye_center, radius, DARK_BLUE, 4, 8, 0);
            }

            cv::rectangle(frame, bbox, DARK_BLUE, 2);
            cv::putText(frame, "Tracking", bbox.tl(), CV_FONT_NORMAL, 0.5, DARK_BLUE);
        } else {
            tracking = false;
            tracker.release();
            tracker = cv::TrackerKCF::create();
        }

        std::stringstream stream;
        stream << "FPS:" << std::setprecision(2) << fps;
        cv::putText(frame, stream.str(), TOP_LEFT_CORNER, CV_FONT_NORMAL, 1, DARK_GREEN);

        cv::imshow("Visor", frame);

        if (cv::waitKey(30) >= 0) {
            break;
        }

        fps = static_cast<int>(cv::getTickFrequency() / (cv::getTickCount() - start));
    }

    tracker.release();
    cap.release();

    std::cout << "End of program\n";
    return 0;
}

cv::Mat computeMatXGradient(const cv::Mat &mat) {
    cv::Mat out(mat.rows, mat.cols, CV_64F);

    for (int y = 0; y < mat.rows; ++y) {
        const auto *Mr = mat.ptr<uchar>(y);
        auto *Or = out.ptr<double>(y);

        Or[0] = Mr[1] - Mr[0];
        for (int x = 1; x < mat.cols - 1; ++x) {
            Or[x] = (Mr[x + 1] - Mr[x - 1]) / 2.0;
        }
        Or[mat.cols - 1] = Mr[mat.cols - 1] - Mr[mat.cols - 2];
    }

    return out;
}