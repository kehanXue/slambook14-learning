cmake_minimum_required(VERSION 3.14)
project(pose_estimation2d2d)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)

find_package(OpenCV 3 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(pose_estimation2d2d pose_estimation2d2d.cpp)
target_link_libraries(pose_estimation2d2d
        ${OpenCV_LIBS})
