cmake_minimum_required(VERSION 3.10)

add_library(UltraFace STATIC ultraface.cpp)

target_include_directories(
    UltraFace PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} 
    ${CMAKE_SOURCE_DIR}/3rdparty/ncnn/src 
    ${CMAKE_BINARY_DIR}/ncnn_build/src 
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/utils
)

target_link_libraries(UltraFace PRIVATE ncnn ${OpenCV_LIBS})
