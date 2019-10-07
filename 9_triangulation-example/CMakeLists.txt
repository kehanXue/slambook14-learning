cmake_minimum_required(VERSION 3.14)
project(triangulation_example)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)

find_package(OpenCV 3 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(triangulation_example triangulation_example.cpp frontend.h frontend.cpp)
target_link_libraries(triangulation_example
        ${OpenCV_LIBS})