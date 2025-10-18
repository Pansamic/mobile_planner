cmake_minimum_required(VERSION 3.25)

project(mobile_planner)

enable_language(C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if ( NOT CMAKE_BUILD_TYPE )
    set(CMAKE_BUILD_TYPE Release CACHE "cmake build type")
endif()

include(cmake/dependencies.cmake)

option(MOBILE_PLANNER_BUILD_TESTS "Build test programs" ON)
option(MOBILE_PLANNER_LOG_MAP OFF)
option(MOBILE_PLANNER_BUILD_SHARED "Build shared library" OFF)

# Use GCC's built-in OpenMP support
find_package(OpenMP REQUIRED)

# Include directories for third-party libraries built with ExternalProject_Add
include_directories(
    ${CMAKE_BINARY_DIR}/_deps/install/include
    ${CMAKE_BINARY_DIR}/_deps/install/include/eigen3
    ${CMAKE_BINARY_DIR}/_deps/install/include/pcl-1.15
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_compile_options(-mavx -mavx2 -fopenmp)

# Source files
set(MOBILE_PLANNER_SOURCES
    src/elevation_map.cpp
    src/grid_map.cpp
    src/point_cloud_process.cpp
)

# Header files
set(MOBILE_PLANNER_HEADERS
    include/mobile_planner/elevation_map.h
    include/mobile_planner/grid_map.h
    include/mobile_planner/mobile_planner.h
    include/mobile_planner/point_cloud_process.h
)

# Create library based on option
if(MOBILE_PLANNER_BUILD_SHARED)
    add_library(mobile_planner SHARED ${MOBILE_PLANNER_SOURCES})
    add_dependencies(mobile_planner eigen pcl)
else()
    add_library(mobile_planner STATIC ${MOBILE_PLANNER_SOURCES})
    add_dependencies(mobile_planner eigen pcl)
endif()

# Add OpenMP support
target_link_libraries(mobile_planner PUBLIC OpenMP::OpenMP_CXX)

# Set include directories for target
target_include_directories(mobile_planner PUBLIC
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)



# Install targets
install(TARGETS mobile_planner
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# Install headers
install(FILES ${MOBILE_PLANNER_HEADERS} 
    DESTINATION include/mobile_planner
)

# Add test executable if tests are enabled
if(MOBILE_PLANNER_BUILD_TESTS)
    add_executable(test_mapping_static test/test_mapping_static.cpp)
    
    # Link PCL libraries
    target_link_libraries(test_mapping_static
        mobile_planner
        ${CMAKE_BINARY_DIR}/_deps/install/lib/libpcl_common.so
        ${CMAKE_BINARY_DIR}/_deps/install/lib/libpcl_io.so
        ${CMAKE_BINARY_DIR}/_deps/install/lib/libyaml-cpp.a
        OpenMP::OpenMP_CXX
    )
endif()