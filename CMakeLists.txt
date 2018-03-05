cmake_minimum_required(VERSION 3.9)
project(EngagementDetector)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED core highgui imgproc dnn)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(EngagementDetector main.cpp)

set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs opencv_dnn)
target_link_libraries(EngagementDetector ${OpenCV_LIBS})