add_library(readerLib STATIC reader.cpp)

target_include_directories(
    readerLib 
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} 
    ${OpenCV_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/utils
)

target_link_libraries(readerLib PRIVATE ${OpenCV_LIBS})
