cmake_minimum_required(VERSION 3.16)
project(C++Basics)

set(CMAKE_CXX_STANDARD 20)

find_package(Torch REQUIRED)

add_executable(C++Basics main.cpp)
target_link_libraries(C++Basics "${TORCH_LIBRARIES}")