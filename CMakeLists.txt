cmake_minimum_required(VERSION 3.2)

project("ctf")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -Ofast -march=native")

SET(CMAKE_BUILD_TYPE "Release")

find_package(OpenCV REQUIRED)
find_package(Boost COMPONENTS filesystem system REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

set(HEADER_FILES src/algorithm.h src/block.h src/superpixel.h src/engine.h src/utils.h src/ctfi.h src/origctf.h)
set(SOURCE_FILES src/algorithm.cpp src/block.cpp src/superpixel.cpp src/engine.cpp src/utils.cpp src/ctfi.cpp src/origctf.cpp)

add_executable(ctfi ${SOURCE_FILES} ${HEADER_FILES} src/main.cpp)
target_link_libraries(ctfi ${OpenCV_LIBS})
target_link_libraries(ctfi ${Boost_LIBRARIES})

add_executable(origctf ${SOURCE_FILES} ${HEADER_FILES} src/main2.cpp)
target_link_libraries(origctf ${OpenCV_LIBS})
target_link_libraries(origctf ${Boost_LIBRARIES})
