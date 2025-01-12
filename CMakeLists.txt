
# set minimal version of cmake
cmake_minimum_required(VERSION 3.18)

# set the project name
project(sl_mlp_2_dsp)

# add source files
file(GLOB SOURCES src/*.cpp)

# add include directories
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/idsp/include)


set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

# add target library
add_library(sl_mlp_2_dsp STATIC ${SOURCES})

