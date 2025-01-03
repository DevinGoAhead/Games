//
// Created by LEI XU on 5/16/19.
//

#ifndef RAYTRACING_INTERSECTION_H
#define RAYTRACING_INTERSECTION_H
#include "Vector.hpp"
#include "Material.hpp"
class Object;
class Sphere;

struct Intersection
{
    Intersection(){
        happened=false;
        coords=Vector3f();
        normal=Vector3f();
        distance= std::numeric_limits<double>::max();
        obj =nullptr;
        m=nullptr; // 材质
    }
    bool happened; // 是否相交
    Vector3f coords; // 相交点的坐标
    Vector3f normal; // 相交点的法线
    double distance; // ray 起点与相交点距离
    Object* obj; // 相交对象
    Material* m; // 相交对象的材质
};
#endif //RAYTRACING_INTERSECTION_H
