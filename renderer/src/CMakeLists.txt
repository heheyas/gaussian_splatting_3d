cmake_minimum_required(VERSION 3.3)

project(renderer)

set(CMAKE_CXX_STANDARD 14)
enable_language(CUDA)
message(STATUS "CUDA enabled")
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  -g -Xcudafe \"--display_error_number --diag_suppress=3057 --diag_suppress=3058 --diag_suppress=3059 --diag_suppress=3060\" -lineinfo -arch=sm_75 ")

set(INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)

find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

pybind11_add_module(renderer-test SHARED ${SOURCES})
target_link_libraries(renderer-test PRIVATE "${TORCH_LIBRARIES}")
target_include_directories(renderer-test PRIVATE "${INCLUDE_DIR}")