cmake_minimum_required(VERSION 2.8)
project(ceres-usage)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++11")

find_package(Ceres REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
        ${CERES_INCLUDE_DIRS}
        ${OpenCV_DIRS}
)

add_executable(curve_fitting ceres_curve_fitting.cpp)

target_link_libraries(curve_fitting
        ${CERES_LIBRARIES}
        ${OpenCV_LIBS})
