cmake_minimum_required(VERSION 3.16)
project(dcgan)

set(CMAKE_CXX_STANDARD 20)

find_package(Torch REQUIRED)

add_executable(dcgan main.cpp)
target_link_libraries(dcgan "${TORCH_LIBRARIES}")