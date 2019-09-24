cmake_minimum_required(VERSION 2.8)
project(useEigen)

set(CMAKE_BUILD_TYPE "Release")

find_package(Eigen3)

add_executable(use_eigen eigen_usage_example.cpp)