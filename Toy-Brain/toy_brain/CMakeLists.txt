# CMakeList.txt : CMake project for toy_brain, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

add_subdirectory ("src")
add_subdirectory ("test")

# Add source to this project's executable.
add_executable (toy_brain 
	"toy_brain_main.cpp" 
	"toy_brain_main.h"
	"src/models.cpp"
	"src/models.h"
	"src/helper_functions.h"
	"src/helper_functions.cpp"
	
)

# TODO: Add tests and install targets if needed.
