add_library(displayLib STATIC display.cpp)

target_include_directories(
    displayLib 
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} 
    ${OpenCV_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/utils
)

target_link_libraries(displayLib PRIVATE ${OpenCV_LIBS})
