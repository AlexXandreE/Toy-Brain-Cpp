# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.8)
set (CMAKE_CXX_STANDARD 17)

if(UNIX OR APPLE)
    message(FATAL_ERROR "Project not yet compatible with non-windows systems")
endif()

project ("Toy-Brain")

# Include sub-projects.
add_subdirectory ("toy_brain")
