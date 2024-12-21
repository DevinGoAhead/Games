#pragma once
#include "Scene.hpp"

struct hit_payload // 负载, 用于记录光线追踪过程中的信息
{
    float tNear; // 起始交点距离, 其实就是初始时, 光线起点到顶点的距离
    uint32_t index;
    Vector2f uv;
    Object* hit_obj;
};

class Renderer
{
public:
    void Render(const Scene& scene);

private:
};