#include <algorithm>
#include <cassert>
#include "BVH.hpp"

BVHAccel::BVHAccel(std::vector<Object*> p, int maxPrimsInNode,
                   SplitMethod splitMethod)
    : maxPrimsInNode(std::min(255, maxPrimsInNode)), splitMethod(splitMethod),
      primitives(std::move(p))
{
    time_t start, stop;
    time(&start);
    if (primitives.empty())
        return;

    root = recursiveBuild(primitives);

    time(&stop);
    double diff = difftime(stop, start);
    int hrs = (int)diff / 3600;
    int mins = ((int)diff / 60) - (hrs * 60);
    int secs = (int)diff - (hrs * 3600) - (mins * 60);

    printf(
        "\rBVH Generation complete: \nTime Taken: %i hrs, %i mins, %i secs\n\n",
        hrs, mins, secs);
}

BVHBuildNode* BVHAccel::recursiveBuild(std::vector<Object*> objects)
{
    BVHBuildNode* node = new BVHBuildNode(); // 创建一个节点

    // Compute bounds of all primitives in BVH node
    Bounds3 bounds; 
	// 创建一个能包围当前对象列表中所有对象的最小的包围盒(AABB)
    for (size_t i = 0; i < objects.size(); ++i)
        bounds = Union(bounds, objects[i]->getBounds()); 
	// 如果划分后的包围盒中只有一个对象(即叶子节点, 停止划分, 初始化node)
    if (objects.size() == 1) {
        // Create leaf _BVHBuildNode_
        node->bounds = objects[0]->getBounds();
        node->object = objects[0];
        node->left = nullptr;
        node->right = nullptr;
        return node; // node 中的包围盒进能包围1个对象
    }
	//如果划分后的baba
    else if (objects.size() == 2) {
        node->left = recursiveBuild(std::vector{objects[0]});
        node->right = recursiveBuild(std::vector{objects[1]});

        node->bounds = Union(node->left->bounds, node->right->bounds);
        return node; // node 中的包围盒仅能包围左右子节点的对象
    }
    else {
        Bounds3 centroidBounds;
		
		// getBounds 函数获取一个能够包围当前对象的最小包围盒
		// Centroid 函数获取当前对象包围盒的重心
		// 最终创建的包围盒的边界对齐的是外围对象的重心, 而不是外围对象的边界
        for (size_t i = 0; i < objects.size(); ++i)
            centroidBounds =
                Union(centroidBounds, objects[i]->getBounds().Centroid());
		// 判断最长的轴
        int dim = centroidBounds.maxExtent();
		// 以重心坐标的 x 或 y 或 z 值 为基准值进行升序排序
        switch (dim) {
        case 0:// x 轴最长
            std::sort(objects.begin(), objects.end(), [](auto f1, auto f2) {
                return f1->getBounds().Centroid().x <
                       f2->getBounds().Centroid().x;
            });
            break;
        case 1:// y 轴最长
            std::sort(objects.begin(), objects.end(), [](auto f1, auto f2) {
                return f1->getBounds().Centroid().y <
                       f2->getBounds().Centroid().y;
            });
            break;
        case 2:// z 轴最长
            std::sort(objects.begin(), objects.end(), [](auto f1, auto f2) {
                return f1->getBounds().Centroid().z <
                       f2->getBounds().Centroid().z;
            });
            break;
        }

        auto beginning = objects.begin(); // 第一个对象的迭代器
        auto middling = objects.begin() + (objects.size() / 2); // 中间对象的迭代器
        auto ending = objects.end(); // 最后一个对象的迭代器

        auto leftshapes = std::vector<Object*>(beginning, middling); // [beginning, middling) 复制到 leftshapes
        auto rightshapes = std::vector<Object*>(middling, ending); // [middling, ending) 复制到 rightshapes

        assert(objects.size() == (leftshapes.size() + rightshapes.size())); // 确保划分后的结果正确

        node->left = recursiveBuild(leftshapes); // 对划分的第一部分的对象继续递归
        node->right = recursiveBuild(rightshapes); // 对划分的另一部分的对象继续递归
		// 当 recursiveBuild(leftshapes) recursiveBuild(rightshapes) 执行完毕后, 两个子节点的包围盒就确定了
		// 此时可以基于两个子节点的包围盒确定父节点的包围盒
		node->bounds = Union(node->left->bounds, node->right->bounds);
    }

	return node;
}

Intersection BVHAccel::Intersect(const Ray& ray) const
{
    Intersection isect;
    if (!root)
        return isect;// 如果 BVH 不存在, 返回一个默认的 Intersection 变量
    isect = BVHAccel::getIntersection(root, ray);
    return isect;
}

Intersection BVHAccel::getIntersection(BVHBuildNode* node, const Ray& ray) const
{
    // TODO Traverse the BVH to find intersection

	if(node == nullptr) return Intersection(); // BVH 不存在(或子树不存在,可能么?)返回默认值
	
	// 叶子节点
	if(node->object != nullptr)
	{
		Intersection intersect;
		// 如果与叶子节点中物体的包围盒相交
		// 进一步判断是否与物体本身相交, 并记录交点信息;
		if(node->bounds.IntersectP(ray)) intersect = node->object->getIntersection(ray);
		return intersect; // 不相交返回初始值, 相交则返回 getIntersection(ray) 中记录过交点的值
	}

	// 非叶子节点
	// 如果与当前节点的包围盒相交
	if(node->bounds.IntersectP(ray))
	{
		Intersection leftIntersect = getIntersection(node->left, ray); // 递归左子节点
		Intersection rightIntersect = getIntersection(node->right, ray); // 递归右子节点
		//判断哪个更近
		return leftIntersect.distance < rightIntersect.distance ? leftIntersect : rightIntersect;
		
		// if(leftIntersect.happened && rightIntersect.happened) // 左右子树均相交
		// {
		// 	return leftIntersect.distance < rightIntersect.distance ? leftIntersect : rightIntersect;
		// }
		// else if(leftIntersect.happened) // 仅左子树相交
		// 	return leftIntersect;
		// else if(rightIntersect.happened) // 仅右子树相交
		// 	return rightIntersect;
		// else
		// 	return rightIntersect;
	}

	// 非叶子节点
	// 如果与当前节点的包围盒不相交
	return Intersection(); // 不与包围盒相交, 直接返回默认值
}