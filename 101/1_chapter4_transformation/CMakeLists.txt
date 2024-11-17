cmake_minimum_required(VERSION 3.10)
project(Rasterizer)

set(CMAKE_CXX_STANDARD 17) # 设置 c++ 编译标准
# set(CMAKE_CXX_STANDARD_REQUIRED True) # 编译器不支持上述标准，cmake 报错

# set(OpenCV_Dir "D:/libraries/opencv_4_10_0/install") # 变量 OpenCV_Dir = OpenCV 的安装（解压）路径(OpenCVConfig.cmake 所在路径)
# find_package(OpenCV REQUIRED PATHS ${OpenCV_Dir}) # OpenCV - PackageName

set(CMAKE_PREFIX_PATH "D:/libraries/opencv_4_10_0/install")
find_package(OpenCV REQUIRED)

set(Eigen_Dir "D:/libraries/eigen_3_4_0/Eigen") # 变量 Eigen_Dir = Eigen 的安装（解压）路径

include_directories(${OpenCV_INCLUDE_DIRS} ${Eigen_Dir}) # 指定头文件的路径
# message(STATUS "~~~~~OpenCV libraries: ${OpenCV_LIBRARIES}~~~~~~")

add_executable(rasterizer rasterizer.cpp triangle.cpp main.cpp)
target_link_libraries(rasterizer ${OpenCV_LIBRARIES})
