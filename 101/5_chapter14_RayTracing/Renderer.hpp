#pragma once
#include "Scene.hpp"

struct hit_payload // 负载, 用于记录光线追踪过程中的信息
{
    float tNear; // 起始交点距离, 其实就是初始时, 光线起点到顶点的距离
	// mesh 的所有三角形的所有顶点都放在一个 vector 中
	// 每 3 个点为一组, 对应一个三角形, 索引 012的点 为第一个三角形, 索引345的点 为第二个三角形...
	// index 则表示hit_payload 对应的交点属于第 index 个三角形
    uint32_t index;
    Vector2f uv; // 重心坐标
    Object* hit_obj;
};

class Renderer
{
public:
    void Render(const Scene& scene);

private:
};