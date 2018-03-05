#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/shape_utils.hpp>

#define CAMERA_NOT_AVAILABLE 1100
#define MODEL_NOT_AVAILABLE 1200
#define CONFIDENCE_LEVEL 0.93

const int width = 300;
const int height = 300;

const static std::string FACE_DNN_PROTO = "models/face/deploy.prototxt.txt";
const static std::string FACE_DNN_MODEL = "models/face/res10_300x300_ssd_iter_140000.caffemodel";
const static auto dark_green = cv::Scalar(50, 180, 0);

int main() {
    auto cap = cv::VideoCapture(0);

    if (!cap.isOpened()) {
        return CAMERA_NOT_AVAILABLE;
    }

    cap.set(CV_CAP_PROP_FRAME_WIDTH, width);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, height);

    auto fps = static_cast<int>(cap.get(CV_CAP_PROP_FPS));

    auto net = cv::dnn::readNetFromCaffe(FACE_DNN_PROTO, FACE_DNN_MODEL);

    if (net.empty()) {
        return MODEL_NOT_AVAILABLE;
    }

    while (true) {
        int64 start = cv::getTickCount();

        cv::Mat frame{};
        if (!cap.read(frame)) {
            std::cerr << "Not able to read frame\n";
            continue;
        }
        cv::flip(frame, frame, 1);

        auto blob = cv::dnn::blobFromImage(frame, 1.0, cv::Size{width, height}, 1.0);
        net.setInput(blob);
        auto res = net.forward();

        res = res.reshape(1, static_cast<int>(res.total() / 7));
        for (int i = 0; i < res.rows; i++) {
            float confidence = res.at<float>(i, 2);

            if (confidence > CONFIDENCE_LEVEL) {
                const auto left = static_cast<int>(res.at<float>(i, 3) * frame.cols);
                const auto top = static_cast<int>(res.at<float>(i, 4) * frame.rows);
                const auto right = static_cast<int>(res.at<float>(i, 5) * frame.cols);
                const auto bottom = static_cast<int>(res.at<float>(i, 6) * frame.rows);

                rectangle(frame, cv::Point(left, top), cv::Point(right, bottom),
                          dark_green);
                std::stringstream stream;
                stream << "Confidence:" << std::setprecision(2) << confidence;
                cv::putText(frame, stream.str(), cv::Point(left, top), CV_FONT_NORMAL, 0.5, dark_green);
            }
        }

        std::stringstream stream;
        stream << "Confidence:" << std::setprecision(2) << fps;
        cv::putText(frame, stream.str(), cv::Point{25, 25}, CV_FONT_NORMAL, 1, dark_green);
        cv::imshow("Camera Test", frame);

        if (cv::waitKey(30) >= 0) {
            break;
        }

        fps = static_cast<int>(cv::getTickFrequency() / (cv::getTickCount() - start));
    }

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