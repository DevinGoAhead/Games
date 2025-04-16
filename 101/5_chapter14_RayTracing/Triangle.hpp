#pragma once

#include "Object.hpp"

#include <cstring>

bool rayTriangleIntersect(const Vector3f& v0, const Vector3f& v1, const Vector3f& v2, const Vector3f& orig,
                          const Vector3f& dir, float& tnear, float& u, float& v)
{
    // TODO: Implement this function that tests whether the triangle
    // that's specified bt v0, v1 and v2 intersects with the ray (whose
    // origin is *orig* and direction is *dir*)
    // Also don't forget to update tnear, u and v.
	Vector3f e1 = v1 - v0;
	Vector3f e2 = v2 - v0;
	Vector3f s = orig - v0;
	Vector3f s1 = crossProduct(dir, e2);
	Vector3f s2 = crossProduct(s, e1);
	float d = dotProduct(s1, e1);
	if(d != 0.f)
	{
		float t = dotProduct(s2, e2) / d;
		float b1 = dotProduct(s1, s) / d;
		float b2 = dotProduct(s2, dir) / d;

		if(t > 0 &&
			(b1 >0 && b1 < 1) && 
			(b2 >0 && b2 < 1) &&
			(1 - b1 - b2 >0 && 1 - b1 - b2 < 1)
		) 
			{
				tnear = t; u = b1; v = b2;
				return true;
			}
	}
	return false;
}

class MeshTriangle : public Object
{
public:
    MeshTriangle(const Vector3f* verts, const uint32_t* vertsIndex, const uint32_t& numTris, const Vector2f* st)
    {
        uint32_t maxIndex = 0;
        for (uint32_t i = 0; i < numTris * 3; ++i)
            if (vertsIndex[i] > maxIndex)
                maxIndex = vertsIndex[i];
        maxIndex += 1; //顶点索引从0开始, 所以加1
        vertices = std::unique_ptr<Vector3f[]>(new Vector3f[maxIndex]);
        memcpy(vertices.get(), verts, sizeof(Vector3f) * maxIndex); // 将顶点坐标, 即 "VBO" 中的顶点数据, 从 verts 指向的空间拷贝到 vertices 指向的空间中
        
		vertexIndex = std::unique_ptr<uint32_t[]>(new uint32_t[numTris * 3]);
        memcpy(vertexIndex.get(), vertsIndex, sizeof(uint32_t) * numTris * 3); // 将顶点坐标索引, 即 "EBO" 中的顶点数据,从vertsIndex 指向的空间拷贝到 vertexIndex 指向的空间中
        numTriangles = numTris;
        stCoordinates = std::unique_ptr<Vector2f[]>(new Vector2f[maxIndex]);
        memcpy(stCoordinates.get(), st, sizeof(Vector2f) * maxIndex); // 将顶点的纹理坐标, 从st指向的空间拷贝到 stCoordinates 指向的空间中
    }
	// tnear, uv, index 都是输出型参数
    bool intersect(const Vector3f& orig, const Vector3f& dir, float& tnear, uint32_t& index,
                   Vector2f& uv) const override
    {
        bool intersect = false;
        for (uint32_t k = 0; k < numTriangles; ++k)
        {
            const Vector3f& v0 = vertices[vertexIndex[k * 3]];
            const Vector3f& v1 = vertices[vertexIndex[k * 3 + 1]];
            const Vector3f& v2 = vertices[vertexIndex[k * 3 + 2]];
            float t, u, v;
			// t, u, v 均为输出型参数, uv 是重心坐标
            if (rayTriangleIntersect(v0, v1, v2, orig, dir, t, u, v) && t < tnear)
            {
                tnear = t;
                uv.x = u;
                uv.y = v;
                index = k;
                intersect |= true;
            }
        }

        return intersect;
    }

	// index, 交点所属三角形在 std::vector<vetex>中的的索引, 每3个点对应一个三角形
	// uv, 重心坐标
	// N, 法向量; st, 纹理坐标, 输出型参数
    void getSurfaceProperties(const Vector3f&, const Vector3f&, const uint32_t& index, const Vector2f& uv, Vector3f& N,
                              Vector2f& st) const override
    {
        const Vector3f& v0 = vertices[vertexIndex[index * 3]];
        const Vector3f& v1 = vertices[vertexIndex[index * 3 + 1]];
        const Vector3f& v2 = vertices[vertexIndex[index * 3 + 2]];
        Vector3f e0 = normalize(v1 - v0);
        Vector3f e1 = normalize(v2 - v1);
        N = normalize(crossProduct(e0, e1)); // 使用两个边向量计算法向量
        const Vector2f& st0 = stCoordinates[vertexIndex[index * 3]];
        const Vector2f& st1 = stCoordinates[vertexIndex[index * 3 + 1]];
        const Vector2f& st2 = stCoordinates[vertexIndex[index * 3 + 2]];
		// 根据重心坐标插值计算得到 hitPoint 的纹理坐标
        st = st0 * (1 - uv.x - uv.y) + st1 * uv.x + st2 * uv.y; 
    }

    Vector3f evalDiffuseColor(const Vector2f& st) const override
    {
        float scale = 5;
		// 式子的计算结果本质是个布尔值, 但是这里用了浮点数, 因为 lerp 需要一个浮点参数
		// 以 x 为例
		// x \in [0,0.1) fmodf 计算式的结果 < 0.5
		// x \in [0.1,0.2) fmodf 计算式的结果 > 0.5
		// x \in [0.2,0.3) fmodf 计算式的结果 < 0.5
		// ...
        float pattern = (fmodf(st.x * scale, 1) > 0.5) ^ (fmodf(st.y * scale, 1) > 0.5);
        return lerp(Vector3f(0.815, 0.235, 0.031), Vector3f(0.937, 0.937, 0.231), pattern);//红色和黄色
    }

    std::unique_ptr<Vector3f[]> vertices;
    uint32_t numTriangles;
    std::unique_ptr<uint32_t[]> vertexIndex;
    std::unique_ptr<Vector2f[]> stCoordinates;
};
