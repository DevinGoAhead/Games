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
    Bounds3()
    {
        double minNum = std::numeric_limits<double>::lowest();
        double maxNum = std::numeric_limits<double>::max();
        pMax = Vector3f(minNum, minNum, minNum);
        pMin = Vector3f(maxNum, maxNum, maxNum);
    }
    Bounds3(const Vector3f p) : pMin(p), pMax(p) {}
    Bounds3(const Vector3f p1, const Vector3f p2)
    {
        pMin = Vector3f(fmin(p1.x, p2.x), fmin(p1.y, p2.y), fmin(p1.z, p2.z));
        pMax = Vector3f(fmax(p1.x, p2.x), fmax(p1.y, p2.y), fmax(p1.z, p2.z));
    }

    Vector3f Diagonal() const { return pMax - pMin; }
    int maxExtent() const
    {
        Vector3f d = Diagonal();
        if (d.x > d.y && d.x > d.z)
            return 0;
        else if (d.y > d.z)
            return 1;
        else
            return 2;
    }

    double SurfaceArea() const
    {
        Vector3f d = Diagonal();
        return 2 * (d.x * d.y + d.x * d.z + d.y * d.z);
    }

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
	//粘贴自上一次代码
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
	/*这一版来自 Assignment 6, 当时是正确的, 但在 Assignment7 却会导致path tracing 无法得到期望的结果, 故放弃!*/ 
	// 找到问题了
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
	//  Cornell box 可能会出现大量的 ray 与包围盒表面刚好相交, 几乎平行的状况
	// 这种情况 由于浮点数误差极可能会 tEnter == tExit, 而 tEnter < tExit 这样的条件可能会判定为不相交
	// 因此这里必须使用 Enter <= tExit 才可以
	// 简单而言, 多数情况下都可以使用 Enter <= tExit 而不是 Enter < tExit 进行判断
	return tExit > 0 && tEnter <= tExit; // 将 tEnter < tExit 改为了 tEnter <= tExit;
}
// 粘贴自上一次代码, 结尾

inline Bounds3 Union(const Bounds3& b1, const Bounds3& b2)
{
    Bounds3 ret;
    ret.pMin = Vector3f::Min(b1.pMin, b2.pMin);
    ret.pMax = Vector3f::Max(b1.pMax, b2.pMax);
    return ret;
}

inline Bounds3 Union(const Bounds3& b, const Vector3f& p)
{
    Bounds3 ret;
    ret.pMin = Vector3f::Min(b.pMin, p);
    ret.pMax = Vector3f::Max(b.pMax, p);
    return ret;
}

#endif // RAYTRACING_BOUNDS3_H
