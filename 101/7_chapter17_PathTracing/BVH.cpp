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
    BVHBuildNode* node = new BVHBuildNode();

    // Compute bounds of all primitives in BVH node
    Bounds3 bounds;
	// 创建一个能包围当前对象列表中所有对象的最小的包围盒(AABB)
    for (int i = 0; i < objects.size(); ++i)
        bounds = Union(bounds, objects[i]->getBounds());
	// 如果划分后的包围盒中只有一个对象(即叶子节点, 停止划分, 初始化node)
    if (objects.size() == 1) {
        // Create leaf _BVHBuildNode_
        node->bounds = objects[0]->getBounds();
        node->object = objects[0];
        node->left = nullptr;
        node->right = nullptr;
        node->area = objects[0]->getArea();
        return node;
    }
	//如果划分后的包围盒中有两个对象
    else if (objects.size() == 2) {
        node->left = recursiveBuild(std::vector{objects[0]});
        node->right = recursiveBuild(std::vector{objects[1]});

        node->bounds = Union(node->left->bounds, node->right->bounds);
        node->area = node->left->area + node->right->area;
        return node;
    }
    else {
        Bounds3 centroidBounds;

		// getBounds 函数获取一个能够包围当前对象的最小包围盒
		// Centroid 函数获取当前对象包围盒的重心
		// 最终创建的包围盒的边界对齐的是外围对象的重心, 而不是外围对象的边界
        for (int i = 0; i < objects.size(); ++i)
            centroidBounds =
                Union(centroidBounds, objects[i]->getBounds().Centroid());
		// 判断最长的轴
        int dim = centroidBounds.maxExtent();
		// 以重心坐标的 x 或 y 或 z 值 为基准值进行升序排序
        switch (dim) {
        case 0:
            std::sort(objects.begin(), objects.end(), [](auto f1, auto f2) {
                return f1->getBounds().Centroid().x <
                       f2->getBounds().Centroid().x;
            });
            break;
        case 1:
            std::sort(objects.begin(), objects.end(), [](auto f1, auto f2) {
                return f1->getBounds().Centroid().y <
                       f2->getBounds().Centroid().y;
            });
            break;
        case 2:
            std::sort(objects.begin(), objects.end(), [](auto f1, auto f2) {
                return f1->getBounds().Centroid().z <
                       f2->getBounds().Centroid().z;
            });
            break;
        }

        auto beginning = objects.begin();// 第一个对象的迭代器
        auto middling = objects.begin() + (objects.size() / 2);// 中间对象的迭代器
        auto ending = objects.end();// 最后一个对象的迭代器

        auto leftshapes = std::vector<Object*>(beginning, middling);// [beginning, middling) 复制到 leftshapes
        auto rightshapes = std::vector<Object*>(middling, ending);// [middling, ending) 复制到 rightshapes

        assert(objects.size() == (leftshapes.size() + rightshapes.size()));// 确保划分后的结果正确

        node->left = recursiveBuild(leftshapes);// 对划分的第一部分的对象继续递归
        node->right = recursiveBuild(rightshapes);// 对划分的另一部分的对象继续递归

		// 当 recursiveBuild(leftshapes) recursiveBuild(rightshapes) 执行完毕后, 两个子节点的包围盒就确定了
		// 此时可以基于两个子节点的包围盒确定父节点的包围盒
		// 本次作业 Assignment 7, 在节点中新增了 左右子节点中 对象 的面积的和
        node->bounds = Union(node->left->bounds, node->right->bounds);
        node->area = node->left->area + node->right->area;
    }

    return node;
}

Intersection BVHAccel::Intersect(const Ray& ray) const
{
    Intersection isect;
    if (!root)
        return isect;
    isect = BVHAccel::getIntersection(root, ray);
    return isect;
}

// 粘贴自上一次作业
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
		
		// 上面左右子节点递归完成后会return Intersection, Intersection 中存储了交点与ray 起点的 distance
		// 判断哪个distance 更近
		return leftIntersect.distance < rightIntersect.distance ? leftIntersect : rightIntersect;
	}

	// 非叶子节点
	// 如果与当前节点的包围盒不相交
	return Intersection(); // 不与包围盒相交, 直接返回默认值
}

void BVHAccel::getSample(BVHBuildNode* node, float p, Intersection &pos, float &pdf){
    // 如果是叶节点
	// 实际 左右子节点也么同时为 nullptr, 要么均不为nullptr, 所以 node->left == nullptr && node->right == nullptr 更合理
	// 但 || 也不会有错误
	if(node->left == nullptr || node->right == nullptr){
        node->object->Sample(pos, pdf); // 调用节点中 对象 的Sample 函数, 本例是Triangle::
        pdf *= node->area; 
        return;
    }
	// 如果 p 小于左子节点, 说明p 落在左段(左子节点), 否则, 右子节点
    if(p < node->left->area) getSample(node->left, p, pos, pdf);
    else getSample(node->right, p - node->left->area, pos, pdf); // 因为右子节点存储的另一段的面积, 而不是累计面积
}

// pos pdf 继续传递
void BVHAccel::Sample(Intersection &pos, float &pdf){
	// 和 Scene 原理相同, 生产一个(0, area)的随机数
	// 这里 area 根节点中的area, 是当前 单个object, 也就是 单个 mesh(多个 triangle 构成) 的总面积
	// 这里同样可以在逻辑上认为object 的总的triangle是有分段, 但边界连续的区域
    float p = std::sqrt(get_random_float()) * root->area;
    getSample(root, p, pos, pdf); // 把 p 和 总的 area 随机数传递过去
    pdf /= root->area;
}