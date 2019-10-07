cmake_minimum_required(VERSION 2.8)
project(useOpencv)

add_compile_options(-std=c++11)
set(CMAKE_BUILD_TYPE "Release")

find_package(OpenCV 3 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(opencv_example opencv_example.cpp)

target_link_libraries(opencv_example ${OpenCV_LIBS})
