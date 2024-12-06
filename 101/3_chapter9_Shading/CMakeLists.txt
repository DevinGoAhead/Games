cmake_minimum_required(VERSION 3.10)
project(RASTERIZER)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_PREFIX_PATH "D:/libraries/opencv_4_10_0/install")
find_package(OpenCV REQUIRED)

set(Eigen_Dir "D:/libraries/eigen_3_4_0/Eigen") # 变量 Eigen_Dir = Eigen 的安装（解压）路径

add_compile_options(-g)

include_directories(${OpenCV_INCLUDE_DIRS} ${Eigen_Dir}) # 指定头文件的路径

add_executable(Rasterizer main.cpp rasterizer.hpp rasterizer.cpp global.hpp Triangle.hpp Triangle.cpp Texture.hpp Texture.cpp Shader.hpp OBJ_Loader.h)
target_link_libraries(Rasterizer ${OpenCV_LIBRARIES})
#target_compile_options(Rasterizer PUBLIC -Wall -Wextra -pedantic)