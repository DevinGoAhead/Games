#pragma once

#include "BVH.hpp"
#include "Intersection.hpp"
#include "Material.hpp"
#include "OBJ_Loader.hpp"
#include "Object.hpp"
#include "Triangle.hpp"
#include <cassert>
#include <array>

bool rayTriangleIntersect(const Vector3f& v0, const Vector3f& v1,
                          const Vector3f& v2, const Vector3f& orig,
                          const Vector3f& dir, float& tnear, float& u, float& v)
{
    Vector3f edge1 = v1 - v0;
    Vector3f edge2 = v2 - v0;
    Vector3f pvec = crossProduct(dir, edge2);
    float det = dotProduct(edge1, pvec);
    if (det == 0 || det < 0)
        return false;

    Vector3f tvec = orig - v0;
    u = dotProduct(tvec, pvec);
    if (u < 0 || u > det)
        return false;

    Vector3f qvec = crossProduct(tvec, edge1);
    v = dotProduct(dir, qvec);
    if (v < 0 || u + v > det)
        return false;

    float invDet = 1 / det;

    tnear = dotProduct(edge2, qvec) * invDet;
    u *= invDet;
    v *= invDet;

    return true;
}

class Triangle : public Object
{
public:
    Vector3f v0, v1, v2; // vertices A, B ,C , counter-clockwise order // 顺时针
    Vector3f e1, e2;     // 2 edges v1-v0, v2-v0; // 两个边
    Vector3f t0, t1, t2; // texture coords // 3个顶点对应的纹理坐标
    Vector3f normal; // 面法线
    Material* m;// 材质

    Triangle(Vector3f _v0, Vector3f _v1, Vector3f _v2, Material* _m = nullptr)
        : v0(_v0), v1(_v1), v2(_v2), m(_m)
    {
        e1 = v1 - v0;
        e2 = v2 - v0;
        normal = normalize(crossProduct(e1, e2));// 单位化
    }

    bool intersect(const Ray& ray) override;
    bool intersect(const Ray& ray, float& tnear,
                   uint32_t& index) const override;
    Intersection getIntersection(Ray ray) override;
    void getSurfaceProperties(const Vector3f& P, const Vector3f& I,
                              const uint32_t& index, const Vector2f& uv,
                              Vector3f& N, Vector2f& st) const override
    {
        N = normal;// 只确定了法线, N 为输出型参数
        //        throw std::runtime_error("triangle::getSurfaceProperties not
        //        implemented.");
    }
    Vector3f evalDiffuseColor(const Vector2f&) const override;
    Bounds3 getBounds() override;
};

class MeshTriangle : public Object
{
public:
   // 这个函数完成了 三件 工作, 详见注释
    MeshTriangle(const std::string& filename) // filename - 路径名称
    {
        objl::Loader loader; // 模型加载器
        loader.LoadFile(filename);
		// LoadedMeshes的类型是 std::vector<Mesh>
		// 要求 size == 1, 说明本例用到的模型是一个整体的mesh, 原始模型的所有的顶点数据都存放在一个mesh 中, 这和之前的奶牛是一样的
        assert(loader.LoadedMeshes.size() == 1); 
        auto mesh = loader.LoadedMeshes[0]; // 将模型的mesh 复制一份
		// 模型包围盒的左下后点, 初始值为正无穷大, 当第一次使用min与新点比较时, min_vert 总是会被被更新为新点
        Vector3f min_vert = Vector3f{std::numeric_limits<float>::infinity(),
                                     std::numeric_limits<float>::infinity(),
                                     std::numeric_limits<float>::infinity()};
        // 模型包围盒的右上前点(也不绝对, 这取决于坐标系的定义, 不过这两个点总能完全确定下来这个包围盒)
		// 初始值为负无穷大, 当第一次使用 max 与新点比较时, max_vert 总是会被被更新为新点
		Vector3f max_vert = Vector3f{-std::numeric_limits<float>::infinity(),
                                     -std::numeric_limits<float>::infinity(),
                                     -std::numeric_limits<float>::infinity()};
		// Vertices 的类型是 std::vector<Vertex>, Vertices 是 Mesh 中的成员, Vertices 中的元素为 struct Vertex
		// Vertex 的成员是 Vector3 Position, Vector3 Normal, Vector2 TextureCoordinate
		// 在这里将依次遍历顶点 Vertex, 一次遍历3个
        for (size_t i = 0; i < mesh.Vertices.size(); i += 3) {
            std::array<Vector3f, 3> face_vertices; // 一个包含3个顶点的array
            for (int j = 0; j < 3; j++) {
                auto vert = Vector3f(mesh.Vertices[i + j].Position.X,
                                     mesh.Vertices[i + j].Position.Y,
                                     mesh.Vertices[i + j].Position.Z) *
                            60.f; // obj 中的文件可能是经过归一化的, 比较小, 这里需要根据场景进行放大, 如果放大倍数不合适, 还需要调整
                face_vertices[j] = vert;
				// 第一次更新的效果是  min_vert == max_vert == 第一次更新的点
                min_vert = Vector3f(std::min(min_vert.x, vert.x),
                                    std::min(min_vert.y, vert.y),
                                    std::min(min_vert.z, vert.z));
                max_vert = Vector3f(std::max(max_vert.x, vert.x),
                                    std::max(max_vert.y, vert.y),
                                    std::max(max_vert.z, vert.z));
            }

            auto new_mat =
                new Material(MaterialType::DIFFUSE_AND_GLOSSY,
                             Vector3f(0.5, 0.5, 0.5), Vector3f(0, 0, 0));// 参数分别为 glossy 材质(不会折射和高光反射), 颜色(灰色), 材质本身不会发光
            new_mat->Kd = 0.6; // 漫反射项系数
            new_mat->Ks = 0.0; // 高光项系数 0
            new_mat->specularExponent = 0; // 高光指数 0

			// 1. 将所有的mesh 中的三角形 push 到 triangles 中
			// triangles是 class MeshTriangle 的成员, triangles 的类型是 std::vector<Triangle>, 
            triangles.emplace_back(face_vertices[0], face_vertices[1],
                                   face_vertices[2], new_mat);// 自动构造,然后 push_back
        }
		// 2. 为 mesh 构建了一个 包围盒(AABB)
		// bounding_box 是 MeshTriangle 的成员
		// 每遍历一个三角形,  bounding_box 都会更新一次
        bounding_box = Bounds3(min_vert, max_vert);

		
        std::vector<Object*> ptrs; // vector 的元素为 基类指针, 指向的对象为 mesh 中的 triangle
        for (auto& tri : triangles)
            ptrs.push_back(&tri); // 将 triangles 中所有 triangle 的指针 push 到 ptrs 中
		
		// 3. 使用 ptrs 构造 BVH
		// bvh 是 MeshTriangle 的成员
        bvh = new BVHAccel(ptrs); 
    }

    bool intersect(const Ray& ray) { return true; }

    bool intersect(const Ray& ray, float& tnear, uint32_t& index) const
    {
        bool intersect = false;
        for (uint32_t k = 0; k < numTriangles; ++k) {
            const Vector3f& v0 = vertices[vertexIndex[k * 3]];
            const Vector3f& v1 = vertices[vertexIndex[k * 3 + 1]];
            const Vector3f& v2 = vertices[vertexIndex[k * 3 + 2]];
            float t, u, v;
            if (rayTriangleIntersect(v0, v1, v2, ray.origin, ray.direction, t,
                                     u, v) &&
                t < tnear) {
                tnear = t;
                index = k;
                intersect |= true;
            }
        }

        return intersect;
    }

    Bounds3 getBounds() { return bounding_box; }

    void getSurfaceProperties(const Vector3f& P, const Vector3f& I,
                              const uint32_t& index, const Vector2f& uv,
                              Vector3f& N, Vector2f& st) const
    {
        const Vector3f& v0 = vertices[vertexIndex[index * 3]];
        const Vector3f& v1 = vertices[vertexIndex[index * 3 + 1]];
        const Vector3f& v2 = vertices[vertexIndex[index * 3 + 2]];
        Vector3f e0 = normalize(v1 - v0);
        Vector3f e1 = normalize(v2 - v1);
        N = normalize(crossProduct(e0, e1));
        const Vector2f& st0 = stCoordinates[vertexIndex[index * 3]];
        const Vector2f& st1 = stCoordinates[vertexIndex[index * 3 + 1]];
        const Vector2f& st2 = stCoordinates[vertexIndex[index * 3 + 2]];
        st = st0 * (1 - uv.x - uv.y) + st1 * uv.x + st2 * uv.y;
    }

    Vector3f evalDiffuseColor(const Vector2f& st) const
    {
        float scale = 5;
        float pattern =
            (fmodf(st.x * scale, 1) > 0.5) ^ (fmodf(st.y * scale, 1) > 0.5);
        return lerp(Vector3f(0.815, 0.235, 0.031),
                    Vector3f(0.937, 0.937, 0.231), pattern);
    }

    Intersection getIntersection(Ray ray)
    {
        Intersection intersec;

        if (bvh) {
            intersec = bvh->Intersect(ray);
        }

        return intersec;
    }

    Bounds3 bounding_box;
    std::unique_ptr<Vector3f[]> vertices;
    uint32_t numTriangles;
    std::unique_ptr<uint32_t[]> vertexIndex;
    std::unique_ptr<Vector2f[]> stCoordinates;

    std::vector<Triangle> triangles;

    BVHAccel* bvh;

    Material* m;
};

inline bool Triangle::intersect(const Ray& ray) { return true; }
inline bool Triangle::intersect(const Ray& ray, float& tnear,
                                uint32_t& index) const
{
    return false;
}

// 定义一个能够包围v0 v1 v2 的最小的包围盒(AABB)
inline Bounds3 Triangle::getBounds() { return Union(Bounds3(v0, v1), v2); }

inline Intersection Triangle::getIntersection(Ray ray)
{
    Intersection inter;

    if (dotProduct(ray.direction, normal) > 0)
        return inter;
    double u, v, t_tmp = 0;
    Vector3f pvec = crossProduct(ray.direction, e2);//s1
    double det = dotProduct(e1, pvec);
    if (fabs(det) < EPSILON)
        return inter;

    double det_inv = 1. / det;
    Vector3f tvec = ray.origin - v0;//s
    u = dotProduct(tvec, pvec) * det_inv;//b1
    if (u < 0 || u > 1)
        return inter;
    Vector3f qvec = crossProduct(tvec, e1);
    v = dotProduct(ray.direction, qvec) * det_inv;//b2
    if (v < 0 || u + v > 1)
        return inter;
    t_tmp = dotProduct(e2, qvec) * det_inv;//t

    // TODO find ray triangle intersection
	if(t_tmp < 0) {return inter;}
	
	inter.happened = true;
	//inter.coords = ( 1 - u - v) * v0 + u * v1 + v * v2;
	inter.coords = ray(t_tmp); // O + t * d 计算交点坐标, 而不是 重心插值
	inter.normal =  normal; // 直接使用面法线作为点(像素的)法线, 最终的效果是 flat shading
	inter.distance = t_tmp;
	inter.obj = this;
	inter.m = m;
	return inter;
}

inline Vector3f Triangle::evalDiffuseColor(const Vector2f&) const
{
    return Vector3f(0.5, 0.5, 0.5);
}
