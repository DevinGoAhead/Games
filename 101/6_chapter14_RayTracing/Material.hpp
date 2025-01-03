//
// Created by LEI XU on 5/16/19.
//

#ifndef RAYTRACING_MATERIAL_H
#define RAYTRACING_MATERIAL_H

#include "Vector.hpp"

enum MaterialType { DIFFUSE_AND_GLOSSY, REFLECTION_AND_REFRACTION, REFLECTION };

class Material{
public:
    MaterialType m_type;
    Vector3f m_color;
    Vector3f m_emission;// 材质的自发光颜色或强度
    float ior;
    float Kd, Ks;
    float specularExponent;// 高光指数
    //Texture tex;

    inline Material(MaterialType t=DIFFUSE_AND_GLOSSY, Vector3f c=Vector3f(1,1,1), Vector3f e=Vector3f(0,0,0));
    inline MaterialType getType();
    inline Vector3f getColor();// 获取材质的颜色
    inline Vector3f getColorAt(double u, double v);
    inline Vector3f getEmission();


};

Material::Material(MaterialType t, Vector3f c, Vector3f e){
    m_type = t;
    m_color = c;
    m_emission = e;
}

MaterialType Material::getType(){return m_type;}
Vector3f Material::getColor(){return m_color;}
Vector3f Material::getEmission() {return m_emission;}

Vector3f Material::getColorAt(double u, double v) {
    return Vector3f();
}
#endif //RAYTRACING_MATERIAL_H
