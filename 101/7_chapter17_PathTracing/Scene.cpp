//
// Created by Göksu Güvendiren on 2019-05-14.
//

#include "Scene.hpp"


void Scene::buildBVH() {
    printf(" - Generating BVH...\n\n");
    this->bvh = new BVHAccel(objects, 1, BVHAccel::SplitMethod::NAIVE);
}

Intersection Scene::intersect(const Ray &ray) const
{
    return this->bvh->Intersect(ray); // 调用 BVHAccel::Intersect
}

// 这里假设有多个光源, 目的是通过伪随机采样, 确定一个采样光源
// pos 和 pdf 都没有用到, 仅仅是传递到了 objects[k]->Sample(pos, pdf), 本例是调用 MeshTriangle::Sample 函数
// pos 获取了随机采样点的法线和坐标, pdf 的计算逻辑还是不太清楚???
void Scene::sampleLight(Intersection &pos, float &pdf) const
{
    float emit_area_sum = 0;
	// 遍历场景中的object, 找出 "发光的物体", 计算总的 area
    for (uint32_t k = 0; k < objects.size(); ++k) {
        if (objects[k]->hasEmit()){
            emit_area_sum += objects[k]->getArea();
        }
    }
	// 生产一个(0, _area_sum) 的随机数
    float p = get_random_float() * emit_area_sum;
    emit_area_sum = 0;
	// 这里可以在逻辑上认为总的光源是有分段, 但边界连续的区域, 然后依次遍历, 求遍历得的总面积, 即可确定p 落在哪个光源段内
    for (uint32_t k = 0; k < objects.size(); ++k) {
        if (objects[k]->hasEmit()){
            emit_area_sum += objects[k]->getArea();
            if (p <= emit_area_sum){
                objects[k]->Sample(pos, pdf);
                break;
            }
        }
    }
}

bool Scene::trace(
        const Ray &ray,
        const std::vector<Object*> &objects,
        float &tNear, uint32_t &index, Object **hitObject)
{
    *hitObject = nullptr;
    for (uint32_t k = 0; k < objects.size(); ++k) {
        float tNearK = kInfinity;
        uint32_t indexK;
        Vector2f uvK;
        if (objects[k]->intersect(ray, tNearK, indexK) && tNearK < tNear) {
            *hitObject = objects[k];
            tNear = tNearK;
            index = indexK;
        }
    }


    return (*hitObject != nullptr);
}

Vector3f w_any(0.f);
Vector3f Scene::castRay(const Ray &ray, int depth) const
{
	// 求交
	// 依次调用 Scene::intersect, BVHAccel::Intersect, BVHAccel::getIntersection, Bounds3::IntersectP, Triangle::getIntersection
	Intersection interE2P = intersect(ray); // 从 eye -> Scene 中的 Object
	
	// 获取与 object 交点 P点的各种属性信息
	Vector3f p_cord = interE2P.coords;
	Material* p_m = interE2P.m;
	Vector3f p_nor = interE2P.normal;
	
	if(!interE2P.happened) return this->backgroundColor; // 如果 hit noting, 表示 hit background only, 所以 return backgroundColor

	Vector3f L_dir = 0; // 直接光照贡献, 初始值为0
	Vector3f L_indir = 0; // 间接光照贡献, 初始值为0
	
	/*** direct light ***/
	// 随机采样光源
	float pdf_light; Intersection samp_pos_light;
	sampleLight(samp_pos_light, pdf_light); // 获得光源随机采样点的坐标, 法线, pdf, 光强
	Vector3f light_cord = samp_pos_light.coords;
	Vector3f light_nor = samp_pos_light.normal;
	Vector3f light_emit = samp_pos_light.emit;

	// 阴影测试, p 点与 采样点 之间是否有遮挡
	Vector3f p2light = light_cord - p_cord; // Vector3f p -> light
	float distance_p2light = p2light.norm(); // 采样点与P 点距离

	Vector3f p2light_n = normalize(p2light);// Vector3f p -> light, 归一化

	Ray ray_test(p_cord + EPSILON * p_nor, p2light_n); // ray p -> light
	//Ray ray_test(p_deviation, p2light_n); // ray p -> light
	Intersection inter_test = intersect(ray_test); // 从 object -> object
	float distance_test = (inter_test.coords - p_cord).norm(); // P 点与 可能的障碍物的距离
	if(std::abs(distance_test - distance_p2light) < 0.01)// 说明无障碍物
		L_dir = light_emit * p_m->eval(w_any, p2light_n, p_nor) *
			 dotProduct(p2light_n, p_nor) * dotProduct(-p2light_n, light_nor) / (distance_p2light * distance_p2light * pdf_light);

	/*** other reflectors ***/
	// if(get_random_float() > RussianRoulette) 认为贡献为0, 直接忽略
	if(get_random_float() < RussianRoulette)
	{
		// 从 p 点随机采样出射方向 wo
		Vector3f p_wo = normalize(p_m->sample(w_any, p_nor));
		Ray ray_p_wo(p_cord + EPSILON * p_nor, p_wo); //  p_cord + EPSILON ?
		Intersection interP2Q = intersect(ray_p_wo); // 从 object p -> object q
		// 获取相交object q 的交点的各种属性信息
		Material *q_m = interP2Q.m;   // q点材质
		
		// direct light 中的 hit nothing, 表示直接逃逸了, 可认为 hit background, 因此 return background
		// 1 如果这里 hit nothing, 则说明这条路径贡献为0, return 0, 所以这里可以忽略

		// 2 如果 hit a light, 在 direct light 中已经考虑了, 所以这里忽略
		
		// 2.1 hit a non-emitting object
		// 说明有间接光可以找到 p 点
		if (interP2Q.happened && !q_m->hasEmission()) 
		{
			float pdf_p = p_m->pdf(w_any, p_wo, p_nor); // p 点 pdf
			if (pdf_p > EPSILON) // pdf_p != 0
				L_indir = castRay(ray_p_wo, depth) * p_m->eval(w_any, p_wo, p_nor) * (dotProduct(p_wo, p_nor)) / pdf_p / RussianRoulette;
		}
	}
	return p_m->getEmission() + L_dir + L_indir;
}