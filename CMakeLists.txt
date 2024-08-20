cmake_minimum_required(VERSION 3.0)

project(ImgObjectiveFunction LANGUAGES CXX CUDA) 

find_package(PNG REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(ImgObjectiveFunction main.cu utils/pngio.h utils/pngio.cpp)

target_link_libraries(ImgObjectiveFunction ${PNG_LIBRARY})
