cmake_minimum_required(VERSION 3.13)
project(RAYTRANCING)

# set(CMAKE_CXX_STANDARD 17) # 当整个项目需要统一标准时, 可以用这个

#set(CMAKE_PREFIX_PATH "D:/libraries/opencv_4_10_0/install")
#find_package(OpenCV REQUIRED)

#set(Eigen_Dir "D:/libraries/eigen_3_4_0/Eigen") # 变量 Eigen_Dir = Eigen 的安装（解压）路径

#include_directories(${OpenCV_INCLUDE_DIRS} ${Eigen_Dir}) # 指定头文件的路径

add_executable(RayTracing main.cpp Object.hpp Vector.hpp Sphere.hpp global.hpp Triangle.hpp Scene.cpp Scene.hpp Light.hpp Renderer.cpp)

#target_compile_options(RayTracing PUBLIC -g -Wall -Wextra -pedantic -Wshadow -Wreturn-type -fsanitize=undefined)
target_compile_options(RayTracing PUBLIC -g -Wall -Wextra -pedantic -Wshadow -Wreturn-type)
target_compile_features(RayTracing PUBLIC cxx_std_17) # 更加精细的控制项目的标准,更建议用这个

# target_link_libraries(BezierCurve ${OpenCV_LIBRARIES})# 本节没有用到OpenCV 库

target_link_libraries(RayTracing PUBLIC)