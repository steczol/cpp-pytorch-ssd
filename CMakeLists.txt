cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(cpp_pytorch_ssd)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set_property(GLOBAL PROPERTY CXX_STANDARD 14)

list(APPEND CMAKE_PREFIX_PATH 
    # if not automatically detected, define your paths to 
    # libtorch, 
    # opencv,
    # python
)


find_package(Python3 3.9 EXACT COMPONENTS Development)

find_package(Torch REQUIRED)
find_package(TorchVision REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(OpenCV REQUIRED)

add_executable(traced_inference_example ../traced_inference_example.cpp)
target_include_directories(traced_inference_example PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(traced_inference_example "${TORCH_LIBRARIES}" TorchVision::TorchVision ${OpenCV_LIBS})