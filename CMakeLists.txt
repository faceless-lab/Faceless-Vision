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
find_package(Dlib REQUIRED)

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

    include_directories(/usr/local/include)
    include_directories(/usr/local/opt/openssl/include)
    link_directories(/usr/local/lib)
    link_directories(/usr/local/opt/openssl/lib)
    link_directories(/usr/local/Cellar/libuv/1.11.0/lib)

endif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

#find_package(OpenSSL REQUIRED)
message(STATUS "Using dlib-${Dlib_VERSION}")

add_library(dlib SHARED IMPORTED)

include_directories(${OpenCV_INCLUDE_DIRS} dlib::dlib)
add_executable(
        FaclessVision
        main.cpp
        src/camera/CameraUtils.cpp src/camera/CameraUtils.h
        src/model/ModelUtils.cpp src/model/ModelUtils.h
        src/types/Face.h
        src/utils/Constants.h
        src/types/Eye.h
        src/utils/Helpers.h src/detector/Detector.cpp src/detector/Detector.h)

set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs opencv_dnn opencv_tracking opencv_objdetect)
target_link_libraries(FaclessVision ${OpenCV_LIBS} dlib::dlib ssl uWS)