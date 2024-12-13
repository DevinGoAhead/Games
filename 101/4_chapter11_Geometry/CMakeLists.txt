cmake_minimum_required(VERSION 3.10)
project(GEOMETRY)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_PREFIX_PATH "D:/libraries/opencv_4_10_0/install")
find_package(OpenCV REQUIRED)

set(Eigen_Dir "D:/libraries/eigen_3_4_0/Eigen") # 变量 Eigen_Dir = Eigen 的安装（解压）路径

add_compile_options(-g)

include_directories(${OpenCV_INCLUDE_DIRS} ${Eigen_Dir}) # 指定头文件的路径

add_executable(BezierCurve main.cpp)
target_link_libraries(BezierCurve ${OpenCV_LIBRARIES})