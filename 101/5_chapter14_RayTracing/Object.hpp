#pragma once

#include "Vector.hpp"
#include "global.hpp"

class Object
{
public:
    Object()
        : materialType(DIFFUSE_AND_GLOSSY)
        , ior(1.3)
        , Kd(0.8)
        , Ks(0.2)
        , diffuseColor(0.2) // 默认漫反射颜色为深灰色
        , specularExponent(25)
    {}

    virtual ~Object() = default;

    virtual bool intersect(const Vector3f&, const Vector3f&, float&, uint32_t&, Vector2f&) const = 0;

    virtual void getSurfaceProperties(const Vector3f&, const Vector3f&, const uint32_t&, const Vector2f&, Vector3f&,
                                      Vector2f&) const = 0;

    virtual Vector3f evalDiffuseColor(const Vector2f&) const // eval, 计算漫反射的颜色, 这里仅仅简单返回默认颜色
    {
        return diffuseColor;
    }

    // material properties
    MaterialType materialType;//材质类型
    float ior;//index of refraction, 折射率
    float Kd, Ks;//漫反射项系数, 高光项系数
    Vector3f diffuseColor;//漫反射颜色
    float specularExponent;// 高光项的幂
};
