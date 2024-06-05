cmake_minimum_required(VERSION 3.12)
project(final_project)
find_package(OpenCV REQUIRED)
add_executable(final_project 
                src/card_crop.cpp
                src/headers.h
                src/segment.cpp
                )

target_link_libraries(final_project ${OpenCV_LIBS})
