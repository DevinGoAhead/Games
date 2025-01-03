//
// Created by LEI XU on 5/16/19.
//

#ifndef RAYTRACING_BVH_H
#define RAYTRACING_BVH_H

#include <atomic>
#include <vector>
#include <memory>
#include <ctime>
#include "Object.hpp"
#include "Ray.hpp"
#include "Bounds3.hpp"
#include "Intersection.hpp"
#include "Vector.hpp"

struct BVHBuildNode;
// BVHAccel Forward Declarations
struct BVHPrimitiveInfo;

// BVHAccel Declarations
inline int leafNodes, totalLeafNodes, totalPrimitives, interiorNodes;
class BVHAccel {

public:
    // BVHAccel Public Types
    enum class SplitMethod { NAIVE, SAH };

    // BVHAccel Public Methods
    BVHAccel(std::vector<Object*> p, int maxPrimsInNode = 1, SplitMethod splitMethod = SplitMethod::NAIVE);
    Bounds3 WorldBound() const;
    ~BVHAccel();

    Intersection Intersect(const Ray &ray) const;
    Intersection getIntersection(BVHBuildNode* node, const Ray& ray)const;
    bool IntersectP(const Ray &ray) const;
    BVHBuildNode* root; // 根节点

    // BVHAccel Private Methods
    BVHBuildNode* recursiveBuild(std::vector<Object*>objects);

    // BVHAccel Private Data
    const int maxPrimsInNode;// 限制 BVH 叶子节点中包含的最大图元（primitive）数量, 了平衡 BVH 的深度和叶子节点的复杂性(但是本练习没有使用该条件)
    const SplitMethod splitMethod; // 划分方法, 本练习仅实现了一种方法, 因此这个参数实际也没有用到
    std::vector<Object*> primitives;
};

// BVH 节点
struct BVHBuildNode {
    Bounds3 bounds;//节点对应的包围盒
    BVHBuildNode *left; // 左子节点
    BVHBuildNode *right; // 右子节点
   	// 指向对象的指针, 当叶子节点时, object 指向对象, 且 left right 为 nullptr
	// 本例中 object 指向的对象为 mesh 中的三角形
    Object* object; 

public:
    int splitAxis=0, firstPrimOffset=0, nPrimitives=0;
    // BVHBuildNode Public Methods
    BVHBuildNode(){
        bounds = Bounds3();
        left = nullptr;right = nullptr;
        object = nullptr;
    }
};




#endif //RAYTRACING_BVH_H
