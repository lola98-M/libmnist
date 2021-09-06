cmake_minimum_required(VERSION 3.0)
project(predict)

find_package(Torch REQUIRED PATHS "/home/libtorch/share/cmake/Torch")
include_directories( ${TORCH_LIBRARIES} )
add_executable(pred inf.cpp)
target_link_libraries(pred ${TORCH_LIBRARIES})
# set C++14 to compile PyTorch
set_property(TARGET pred PROPERTY CXX_STANDARD 14)
