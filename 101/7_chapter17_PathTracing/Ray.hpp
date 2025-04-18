//
// Created by LEI XU on 5/16/19.
//

#include <array>

#ifndef RAYTRACING_RAY_H
#define RAYTRACING_RAY_H
#include "Vector.hpp"
struct Ray{
    //Destination = origin + t*direction
    Vector3f origin;// 光线起点
    Vector3f direction, direction_inv;// 光线的方向单位向量和其坐标的倒数 组成的向量
    double t;//transportation time,
    double t_min, t_max;
	std::array<int, 3> dirIsNeg; // Devin: 调整了代码框架, 粘贴自上一次作业

    Ray(const Vector3f& ori, const Vector3f& dir, const double _t = 0.0): origin(ori), direction(dir),t(_t) {
        direction_inv = Vector3f(1./direction.x, 1./direction.y, 1./direction.z);
        t_min = 0.0;
        t_max = std::numeric_limits<double>::max();
		for(int i = 0; i < 3; ++i) {dirIsNeg[i] = dir[i] > 0 ? 1 : 0;} // Devin: 调整了代码框架, 粘贴自上一次作业
    }

    Vector3f operator()(double t) const{return origin+direction*t;}

    friend std::ostream &operator<<(std::ostream& os, const Ray& r){
        os<<"[origin:="<<r.origin<<", direction="<<r.direction<<", time="<< r.t<<"]\n";
        return os;
    }
};
#endif //RAYTRACING_RAY_H
