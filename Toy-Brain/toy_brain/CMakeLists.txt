# CMakeList.txt : CMake project for toy_brain, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.18)

add_subdirectory ("test")

# Build ToyBrain as a library
file( GLOB LIB_SOURCES src/*.cpp src/models/*.cpp )
file( GLOB LIB_HEADERS src/*.h src/models/*.h )
list(APPEND LIB_SOURCES src/models/multilayer_perceptron.cpp)
list(APPEND LIB_HEADERS src/models/multilayer_perceptron.h)
add_library( ToyBrain ${LIB_SOURCES} ${LIB_HEADERS} )


add_executable (perceptron "examples/perceptron.cpp")
add_executable (mnist "examples/mnist.cpp")

target_link_libraries(perceptron ToyBrain)
target_link_libraries(mnist ToyBrain)

set(DATA_SOURCE_DIR "${CMAKE_SOURCE_DIR}/../data")
set(DATA_DEST_DIR "${CMAKE_CURRENT_BINARY_DIR}/data")

function(add_post_build_copy_data target_name)
	add_custom_command(TARGET ${target_name} POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E make_directory "${DATA_DEST_DIR}"
		COMMAND ${CMAKE_COMMAND} -E copy_directory "${DATA_SOURCE_DIR}" "${DATA_DEST_DIR}"
		COMMENT "Copying data directory to ${DATA_DEST_DIR}"
	)
endfunction()

add_post_build_copy_data(perceptron)
add_post_build_copy_data(mnist)


# TODO: Add tests and install targets if needed.
