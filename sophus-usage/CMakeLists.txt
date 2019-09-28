cmake_minimum_required(VERSION 2.8)
project(useSophus)

add_compile_options(-std=c++11)
set(CMAKE_BUILD_TYPE "Release")

find_package(Eigen3 REQUIRED)
find_package(Sophus REQUIRED)


include_directories(
        ${Sophus_INCLUDE_DIR}
        ${EIGEN3_INCLUDE_DIR}
)

add_executable(use_sophus sophus_usage_example.cpp)
target_link_libraries(use_sophus ${Sophus_LIBRARIES})