# CMakeList.txt : CMake project for toy_brain, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

add_subdirectory ( "examples")
add_subdirectory ("src")
add_subdirectory ("test")

# Build ToyBrain as a library
file( GLOB LIB_SOURCES src/*.cpp )
file( GLOB LIB_HEADERS src/*.h )
add_library( ToyBrain ${LIB_SOURCES} ${LIB_HEADERS} )


add_executable (perceptron "examples/perceptron.cpp" ${LIB_HEADERS})

target_link_libraries(perceptron ToyBrain)


# TODO: Add tests and install targets if needed.
