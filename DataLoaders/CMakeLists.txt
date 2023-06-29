cmake_minimum_required(VERSION 3.25)
project(DataLoader CXX)

set(CMAKE_CXX_STANDARD 23)

add_executable(DataLoader ImageData.h)
set_target_properties(DataLoader PROPERTIES LINKER_LANGUAGE CXX)
#include_directories(external)

