//
// Created by LEI XU on 5/16/19.
//

#ifndef RAYTRACING_BOUNDS3_H
#define RAYTRACING_BOUNDS3_H
#include "Ray.hpp"
#include "Vector.hpp"
#include <limits>
#include <array>

class Bounds3
{
  public:
    Vector3f pMin, pMax; // two points to specify the bounding box
	// 以double 类型最大值定义包围盒的左下后点
	// 以 double 类型的最小值定义包围盒的右上前点
	// 确保第一次更新时, 总能将 pMax pMin 更新为新的点
    Bounds3()
    {
        double minNum = std::numeric_limits<double>::lowest(); // double 类型的最小值
        double maxNum = std::numeric_limits<double>::max(); // double 类型的最大值
        pMax = Vector3f(minNum, minNum, minNum);
        pMin = Vector3f(maxNum, maxNum, maxNum);
    }
    Bounds3(const Vector3f p) : pMin(p), pMax(p) {}

	// 分别取p1 p2 的最小的x y z 构成左下后点 和 右上前 点, 构成两个新点, 定义一个能够包围两个点的最小的包围盒(AABB)
    Bounds3(const Vector3f p1, const Vector3f p2)
    {
        pMin = Vector3f(fmin(p1.x, p2.x), fmin(p1.y, p2.y), fmin(p1.z, p2.z));
        pMax = Vector3f(fmax(p1.x, p2.x), fmax(p1.y, p2.y), fmax(p1.z, p2.z));
    }

    Vector3f Diagonal() const { return pMax - pMin; }

	// 找到最长轴
    int maxExtent() const
    {
        Vector3f d = Diagonal();
        if (d.x > d.y && d.x > d.z)
             return 0; // x 最大
        else if (d.y > d.z) //x 一定 <= y
            return 1; // y 最大
        else
            return 2; // z 最大
    }

    double SurfaceArea() const
    {
        Vector3f d = Diagonal();
        return 2 * (d.x * d.y + d.x * d.z + d.y * d.z);
    }

	// 获取包围盒的左下后, 右上前点构成的对角线的中心点, 即包围盒的重心
    Vector3f Centroid() { return 0.5 * pMin + 0.5 * pMax; }
    Bounds3 Intersect(const Bounds3& b)
    {
        return Bounds3(Vector3f(fmax(pMin.x, b.pMin.x), fmax(pMin.y, b.pMin.y),
                                fmax(pMin.z, b.pMin.z)),
                       Vector3f(fmin(pMax.x, b.pMax.x), fmin(pMax.y, b.pMax.y),
                                fmin(pMax.z, b.pMax.z)));
    }

    Vector3f Offset(const Vector3f& p) const
    {
        Vector3f o = p - pMin;
        if (pMax.x > pMin.x)
            o.x /= pMax.x - pMin.x;
        if (pMax.y > pMin.y)
            o.y /= pMax.y - pMin.y;
        if (pMax.z > pMin.z)
            o.z /= pMax.z - pMin.z;
        return o;
    }

    bool Overlaps(const Bounds3& b1, const Bounds3& b2)
    {
        bool x = (b1.pMax.x >= b2.pMin.x) && (b1.pMin.x <= b2.pMax.x);
        bool y = (b1.pMax.y >= b2.pMin.y) && (b1.pMin.y <= b2.pMax.y);
        bool z = (b1.pMax.z >= b2.pMin.z) && (b1.pMin.z <= b2.pMax.z);
        return (x && y && z);
    }

    bool Inside(const Vector3f& p, const Bounds3& b)
    {
        return (p.x >= b.pMin.x && p.x <= b.pMax.x && p.y >= b.pMin.y &&
                p.y <= b.pMax.y && p.z >= b.pMin.z && p.z <= b.pMax.z);
    }
    inline const Vector3f& operator[](int i) const
    {
        return (i == 0) ? pMin : pMax;
    }

    // inline bool IntersectP(const Ray& ray, const Vector3f& invDir,
    //                        const std::array<int, 3>& dirisNeg) const;
   	inline bool IntersectP(const Ray& ray) const; // Devin: 调整了代码框架
};


inline bool Bounds3::IntersectP(const Ray& ray) const // Devin: 调整了代码框架
{
// inline bool Bounds3::IntersectP(const Ray& ray, const Vector3f& invDir,
//                                 const std::array<int, 3>& dirIsNeg) const
// {
    // invDir: ray direction(x,y,z), invDir=(1.0/x,1.0/y,1.0/z), use this because Multiply is faster that Division
    // dirIsNeg: ray direction(x,y,z), dirIsNeg=[int(x>0),int(y>0),int(z>0)], use this to simplify your logic
    // TODO test if ray bound intersects
	double tEnter = -std::numeric_limits<double>::infinity(); // < 0
	double tExit = std::numeric_limits<double>::infinity();// > 0
	for(int i = 0; i < 3; ++i)
	{
		double t1 = (pMin[i] - ray.origin[i]) * ray.direction_inv[i];
		double t2 = (pMax[i] - ray.origin[i]) * ray.direction_inv[i];
		
		if(ray.dirIsNeg[i] == 0) std::swap(t1, t2);

		tEnter = std::max(t1, tEnter); // 第一次一定会更新为 t1
		tExit = std::min(t2, tExit); // 第一次一定会更新为 t2
		if(tEnter > tExit) return false;
	}
	return tExit > 0 && tEnter < tExit;
    // double t1 = 0;
    // double t2 = 0;
    // t1 = (pMin.x - ray.origin.x) * invDir.x;
    // t2 = (pMax.x - ray.origin.x) * invDir.x;
    // double txmin = (dirIsNeg[0]>0)?t1:t2;
    // double txmax = (dirIsNeg[0]>0)?t2:t1;
    // t1 = (pMin.y - ray.origin.y) * invDir.y;
    // t2 = (pMax.y - ray.origin.y) * invDir.y;
    // double tymin = (dirIsNeg[1]>0)?t1:t2;
    // double tymax = (dirIsNeg[1]>0)?t2:t1;
    // t1 = (pMin.z - ray.origin.z) * invDir.z;
    // t2 = (pMax.z - ray.origin.z) * invDir.z;
    // double tzmin = (dirIsNeg[2]>0)?t1:t2;
    // double tzmax = (dirIsNeg[2]>0)?t2:t1;
    
    // if(std::max(std::max(txmin,tymin),tzmin) < std::min(std::min(txmax,tymax),tzmax) && std::min(std::min(txmax,tymax),tzmax))
    // return true;
    // else
    // return false;
}

// 创建一个能包围p1 p2 的最小的包围盒(AABB), 但不改变 b1 b2
inline Bounds3 Union(const Bounds3& b1, const Bounds3& b2)
{
    Bounds3 ret;
    ret.pMin = Vector3f::Min(b1.pMin, b2.pMin);
    ret.pMax = Vector3f::Max(b1.pMax, b2.pMax);
    return ret;
}

// 在包围盒 b 的基础上, 加入点p, 创建新的包围盒, 不改变包围盒b
inline Bounds3 Union(const Bounds3& b, const Vector3f& p)
{
    Bounds3 ret;
    ret.pMin = Vector3f::Min(b.pMin, p);
    ret.pMax = Vector3f::Max(b.pMax, p);
    return ret;
}

#endif // RAYTRACING_BOUNDS3_H
