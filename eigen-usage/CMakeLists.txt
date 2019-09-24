cmake_minimum_required(VERSION 2.8)
project(useEigen)

set(CMAKE_BUILD_TYPE "Release")

find_package(Eigen3 REQUIRED)

include_directories(
        ${EIGEN3_INCLUDE_DIR}
)
add_definitions(${EIGEN3_DEFINITIONS})
add_executable(use_eigen eigen_usage_example.cpp)