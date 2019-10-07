cmake_minimum_required(VERSION 2.8)
project(feature-process)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++11")

find_package(OpenCV 3 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(feature_process feature_process.cpp)
target_link_libraries(feature_process
        ${OpenCV_LIBS})