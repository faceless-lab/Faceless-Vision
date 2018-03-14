cmake_minimum_required(VERSION 3.9)
project(FaclessVision)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_VERBOSE_MAKEFILE ON)

#set(CMAKE_BUILD_TYPE Release)
#
#set(CMAKE_CXX_FLAGS "-Wall -Wextra")
#set(CMAKE_CXX_FLAGS_DEBUG "-g")
#set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(OpenCV REQUIRED core highgui imgproc dnn)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(FaclessVision main.cpp src/camera/CameraUtils.cpp src/camera/CameraUtils.h src/model/ModelUtils.cpp src/model/ModelUtils.h src/types/Face.h)

set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs opencv_dnn opencv_tracking)
target_link_libraries(FaclessVision ${OpenCV_LIBS})