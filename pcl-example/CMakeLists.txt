cmake_minimum_required(VERSION 2.8)
project(pclExample)

set(CMAKE_BUILD_TYPE  Release)
set(CMAKE_CXX_FLAGS "-std=c++11")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(PCL REQUIRED COMPONENT common io)
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(pcl_example pcl_example.cpp)
target_link_libraries(pcl_example ${OpenCV_LIBS} ${PCL_LIBRARIES})
