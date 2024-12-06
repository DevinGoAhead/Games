//
// Created by LEI XU on 4/27/19.
//

#ifndef RASTERIZER_TEXTURE_H
#define RASTERIZER_TEXTURE_H
#include "global.hpp"
#include <Eigen>
#include <opencv2/opencv.hpp>
class Texture{
private:
    cv::Mat image_data;

public:
    Texture(const std::string& name)
    {
        image_data = cv::imread(name); // 从文件 name 中读取图像数据, 保存在 image_data 中
		// OpenCV 默认使用GBR颜色空间, 因此这里将颜色空间从RGB 转换到 GBR
        cv::cvtColor(image_data, image_data, cv::COLOR_RGB2BGR); 
        width = image_data.cols; // 将 image 的 像素列数(宽度) 存储在 width 中
        height = image_data.rows; // 将 image 的 像素行数(高度) 存储在 height 中
    }

    int width, height;

    Eigen::Vector3f getColor(float u, float v)
    {
        auto u_img = u * width; // 将纹理坐标 u 转换为 image 的 x 坐标
        auto v_img = (1 - v) * height; // 将纹理坐标 v 转换为 image 的 y 坐标
		// 获取 image 对应位置的颜色, cv::Vec3b 表示包含 3 个字节 (BGR) 的向量
        auto color = image_data.at<cv::Vec3b>(v_img, u_img); 
        return Eigen::Vector3f(color[0], color[1], color[2]); // 将颜色值转换为 Eigen::Vector3f 类型的向量并返回
    }
};
#endif //RASTERIZER_TEXTURE_H
