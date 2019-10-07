cmake_minimum_required(VERSION 2.8)
project(g2o-usage)

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++11")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_DIRS})

find_package(G2O REQUIRED)
find_package(Eigen3 REQUIRED)
include_directories(
        ${G2O_INCLUDE_DIR}
        ${EIGEN3_INCLUDE_DIR}
)


add_executable(g2o_usage g2o_curve_fitting.cpp)
target_link_libraries( g2o_usage
        ${OpenCV_LIBS}
        ${G2O_CORE_LIBRARY}
        ${G2O_STUFF_LIBRARY}
)

