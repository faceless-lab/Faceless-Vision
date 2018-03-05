cmake_minimum_required(VERSION 3.9)
project(FaclessVision)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED core highgui imgproc dnn)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(FaclessVision main.cpp src/camera/CameraUtils.cpp src/camera/CameraUtils.h src/model/ModelUtils.cpp src/model/ModelUtils.h)

set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs opencv_dnn)
target_link_libraries(FaclessVision ${OpenCV_LIBS})