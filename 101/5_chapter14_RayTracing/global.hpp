#pragma once

#include <cmath>
#include <iostream>
#include <random>

#define M_PI 3.14159265358979323846

constexpr float kInfinity = std::numeric_limits<float>::max(); // 返回 float 类型的最大值

inline float clamp(const float& lo, const float& hi, const float& v) // 把数限制在 特定范围内
{
    return std::max(lo, std::min(hi, v));
}

inline bool solveQuadratic(const float& a, const float& b, const float& c, float& x0, float& x1)
{
    float discr = b * b - 4 * a * c;
    if (discr < 0)
        return false;
    else if (discr == 0)
        x0 = x1 = -0.5 * b / a;
    else
    {
        float q = (b > 0) ? -0.5 * (b + sqrt(discr)) : -0.5 * (b - sqrt(discr));// 避免精度b 与 sqrt(discr) 运算产生精度损失 
        x0 = q / a;
        x1 = c / q;// x_1 * x_2 = a / c
    }
    if (x0 > x1)
        std::swap(x0, x1); // 确保 x_1 > x_0x_0
    return true;
}

enum MaterialType
{
    DIFFUSE_AND_GLOSSY, // 漫反射与光滑明亮, 木制桌面 / 磨砂玻璃
    REFLECTION_AND_REFRACTION, // 反射 与 折射, 玻璃 / 水
    REFLECTION // 反射, 镜子
};

inline float get_random_float()
{
    std::random_device dev; // 生成随机数种子
    std::mt19937 rng(dev()); // 梅森旋转算法的伪随机数生成器, 使用 dev() 生成的种子值进行初始化
    std::uniform_real_distribution<float> dist(0.f, 1.f); // distribution in range [1, 6] // 均匀分布生成器，用于生成指定范围内的浮点数

    return dist(rng);//用 rng 生成一个随机数，并通过 dist 将其转换为 [0, 1) 范围内的浮点数
}

inline void UpdateProgress(float progress)
{
    int barWidth = 70;

    std::cout << "[";
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i)
    {
        if (i < pos)
            std::cout << "=";
        else if (i == pos)
            std::cout << ">";
        else
            std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << " %\r";
    std::cout.flush();
}
