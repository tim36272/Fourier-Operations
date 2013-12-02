cmake_minimum_required(VERSION 2.8)
project( CS474_Assignment_4 )

#redirect OpenCV to OpenGL installation
#set(OpenCV_DIR "/opt/opencv_2.4.7/")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
#add_definitions(${OpenCV_DEFINITIONS})
MESSAGE(STATUS "OpenCV at: ${OpenCV_INCLUDE_DIRS}")

add_executable( part1 part1.cpp )
target_link_libraries( part1 ${OpenCV_LIBS})

add_executable( part2 part2.cpp )
target_link_libraries( part2 ${OpenCV_LIBS} )

add_executable( part3 part3.cpp )
target_link_libraries( part3 ${OpenCV_LIBS} )
