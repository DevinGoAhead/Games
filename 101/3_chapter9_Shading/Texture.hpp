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

	 Eigen::Vector3f getColorBilinear(float u, float v)
	 {
		auto u_img = u * width; // 将纹理坐标 u 转换为 image 的 x 坐标
        auto v_img = (1 - v) * height; // 将纹理坐标 v 转换为 image 的 y 坐标
		// 获取 image 对应位置的颜色, cv::Vec3b 表示包含 3 个字节 (BGR) 的向量
		auto v_img_00 = (int)v_img;
		auto u_img_00 = (int)u_img;
		
		auto v_img_01 = (int)v_img;
		auto u_img_01 = (int)std::min(u_img + 1, (float)(width -1));
		
		auto v_img_10 = (int)std::min(v_img + 1, (float)(height - 1));
		auto u_img_10 = (int)u_img;
		
		auto v_img_11 = (int)std::min(v_img + 1, (float)(height - 1));
		auto u_img_11 = (int)std::min(u_img + 1, (float)(width -1));

        auto color_00 = image_data.at<cv::Vec3b>(v_img_00, u_img_00);
		auto color_01 = image_data.at<cv::Vec3b>(v_img_01, u_img_01);
		auto color_10 = image_data.at<cv::Vec3b>(v_img_10, u_img_10);
		auto color_11 = image_data.at<cv::Vec3b>(v_img_11, u_img_11);
		
		auto ratio_u = (u_img -  (float)u_img_00) / (float)(u_img_01 - u_img_00);
		auto ratio_v = (v_img -  (float)v_img_00) / (float)(v_img_10 - v_img_00);

		auto color_0 = color_00 + ratio_u * (color_01 - color_00);
		auto color_1 = color_00 + ratio_u * (color_11 - color_10);

		auto color = color_0 + ratio_v * (color_1 - color_0);

		return Eigen::Vector3f(color[0], color[1], color[2]);
	 }
};
#endif //RASTERIZER_TEXTURE_H
