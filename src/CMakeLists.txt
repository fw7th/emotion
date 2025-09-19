cmake_minimum_required(VERSION 3.10)
project(EmotionSense)

# Build settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Stronger optimization for Release
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2 -DNDEBUG -Wall")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")

# Set default to Release if none given
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "No build type selected, defaulting to Release")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build: Debug Release RelWithDebInfo MinSizeRel" FORCE)
endif()

# External dependencies
add_subdirectory(3rdparty/ncnn ncnn_build)
find_package(OpenCV REQUIRED)
find_package(OpenMP)

# Debug output
message(STATUS "OpenCV_FOUND: ${OpenCV_FOUND}")
message(STATUS "OpenCV_VERSION: ${OpenCV_VERSION}")
message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")
message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")

# Project modules
add_subdirectory(reader)
add_subdirectory(detection)
add_subdirectory(tracking)
add_subdirectory(emoreg)

add_executable(emotion main.cpp)
target_link_libraries(emotion PRIVATE readerLib UltraFace tracker Emote)
