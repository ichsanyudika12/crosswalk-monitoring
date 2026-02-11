cmake_minimum_required(VERSION 3.10)
project(TTG2025)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# -----------------------------------
# ONNX Runtime setup
# -----------------------------------
if(NOT DEFINED ONNXRUNTIME_DIR)
    set(ONNXRUNTIME_DIR $ENV{ONNXRUNTIME_DIR})
endif()

if(NOT DEFINED ONNXRUNTIME_DIR)
    message(FATAL_ERROR "ONNX Runtime directory not specified! Set ONNXRUNTIME_DIR environment variable.")
else()
    message(STATUS "ONNXRUNTIME_DIR: ${ONNXRUNTIME_DIR}")
endif()

# -----------------------------------
# Find required packages
# -----------------------------------
find_package(OpenCV REQUIRED)
find_package(ZED 3 REQUIRED)
find_package(CUDA REQUIRED)

# -----------------------------------
# Include directories
# -----------------------------------
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    /usr/local/cuda/include
    "${ONNXRUNTIME_DIR}/include"
    "/usr/local/zed/include"
    "${OpenCV_INCLUDE_DIRS}"
)

# -----------------------------------
# Link directories
# -----------------------------------
link_directories(
    "/usr/local/zed/lib"
    "/usr/local/cuda/lib64"
    "${ONNXRUNTIME_DIR}/lib"
    "/usr/local/lib"                        
)

# -----------------------------------
# Define ZED + CUDA libraries
# -----------------------------------
set(ZED_LIBS ${ZED_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_LIBRARIES})

# -----------------------------------
# Add executable
# -----------------------------------
add_executable(main
    src/realtime.cpp
    src/yolov8Predictor.cpp
    src/utils.cpp
)

# -----------------------------------
# Link libraries
# -----------------------------------
target_link_libraries(main
    ${OpenCV_LIBS}
    ${ZED_LIBS}
    "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so"
)

# -----------------------------------
# Linux-specific flags
# -----------------------------------
if(UNIX AND NOT APPLE)
    add_definitions(-Wno-format-extra-args)
    target_link_libraries(main pthread X11)
endif()
