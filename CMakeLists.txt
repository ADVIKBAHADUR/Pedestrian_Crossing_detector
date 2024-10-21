cmake_minimum_required(VERSION 3.10)
project(myApplication)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Add the executable
add_executable(myapp Crossings.cpp)

# Link OpenCV libraries
target_link_libraries(myapp ${OpenCV_LIBS})
