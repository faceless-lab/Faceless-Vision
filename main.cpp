#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/dnn/shape_utils.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing/shape_predictor.h>
#include <dlib/image_processing.h>
#include <dlib/opencv.h>
#include <uWS/uWS.h>
#include "src/camera/CameraUtils.h"
#include "src/model/ModelUtils.h"
#include "src/types/Face.h"
#include "src/utils/Constants.h"
#include "src/utils/Helpers.h"
#include "src/detector/Detector.h"

#define TOP_LEFT_CORNER cv::Point(25, 25)
#define DARK_GREEN cv::Scalar(50, 180, 0)
#define DARK_BLUE cv::Scalar(180, 50, 0)

const int width = 300;
const int height = 300;


int main() {
    uWS::Hub hub;

    hub.onConnection([&hub](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
        std::cout << "Connected!!!\n";
    });

    hub.onDisconnection([&hub](uWS::WebSocket<uWS::SERVER> ws, int code, char *message, size_t length) {
        ws.close();
        std::cout << "Disconnected\n";
    });

    hub.listen(8787);
    hub.run();

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

    auto face_detector = dlib::get_frontal_face_detector();

    dlib::shape_predictor shape_predictor;
    dlib::deserialize(EYE_DLIB_PREDICTOR_MODEL) >> shape_predictor;

    // TODO: MultiTracker needed. Seems we need a custom implementation.
//    auto tracker = cv::TrackerKCF::create();

    flv::detect::Detector detector("KCF");

//    cv::Rect2d bbox;
    bool tracking = false;
    while (true) {
        int64 start = cv::getTickCount();

        cv::Mat frame{};
        if (!cap.read(frame)) {
            std::cerr << "Not able to read frame\n";
            continue;
        }
        cv::flip(frame, frame, 1);

        cv::Mat gray{};
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        if (!tracking) {
            cv::Mat blob;
            flv::ModelUtils::blob_from_image(frame, blob, cv::Size(width, height));

            net.setInput(blob);
            auto res = net.forward();

            res = res.reshape(1, static_cast<int>(res.total() / 7));

            std::vector<cv::Rect2d> faces{};

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

//                    bbox = face.get_bbox();
                    faces.push_back(face.get_bbox());

//                    tracker->init(gray, bbox);

//                    std::cout << "Find a face; attempting to track\n";
                }
            }

            detector.track(gray, faces);

            blob.release();
            res.release();
        }

        // TODO: update to track multiple faces
//        if (tracker->update(gray, bbox)) {
        if (detector.update(gray)) {
            tracking = true;

            for (auto bbox : detector.get_bboxes()) {
                if (is_inside(bbox, gray)) {
                    auto face = gray(bbox);

                    std::vector<cv::Rect> eyes;

                    eye_csf.detectMultiScale(face, eyes, 1.05, 6, 0, cv::Size(30, 30), cv::Size(80, 80));

                    // dlib::array2d<dlib::bgr_pixel> dl_img;
                    // dlib::assign_image(dl_img, dlib::cv_image<dlib::bgr_pixel>(face));

                    // dlib::rectangle rect;
                    // rect.set_left(static_cast<long>(bbox.x));
                    // rect.set_top(static_cast<long>(bbox.y));
                    // rect.set_right(static_cast<long>(bbox.x + bbox.width));
                    // rect.set_bottom(static_cast<long>(bbox.y + bbox.height));

                    // // TODO: parse the shapes. Replace Face detection with dlib for speeding up the computation.
                    // dlib::full_object_detection shape = shape_predictor(dl_img, rect);

                    for (const auto &eye : eyes) {
                        cv::Point eye_center(static_cast<int>(bbox.x + eye.x + eye.width / 2),
                                             static_cast<int>(bbox.y + eye.y + eye.height / 2));
                        int radius = cvRound((eye.width + eye.height) * 0.25);
                        cv::circle(frame, eye_center, radius, DARK_BLUE, 4, 8, 0);
                    }
                }

                cv::rectangle(frame, bbox, DARK_BLUE, 2);
                cv::putText(frame, "Tracking", bbox.tl(), CV_FONT_NORMAL, 0.5, DARK_BLUE);
            }
        } else {
            tracking = false;
            detector.cleanup();
//            tracker.release();
//            tracker = cv::TrackerKCF::create();
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

    detector.cleanup();
//    tracker.release();
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