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

// Vector3f Scene::MyShade(Intersection intersection, Vector3f wo)const // Devin for Assignment 7
// {
// 	Vector3f L_dir = 0; // 直接光照贡献, 初始值为0
// 	Vector3f L_indir = 0; // 间接光照贡献, 初始值为0
	
// 	Vector3f p_cord = intersection.coords;
// 	Material* p_m = intersection.m;
// 	Vector3f p_nor = intersection.normal;
	
// 	/*** direct light ***/
// 	// 随机采样光源
// 	float pdf_light; Intersection samp_pos_light;
// 	sampleLight(samp_pos_light, pdf_light); // 获得随机采样点的坐标, 法线, pdf
// 	Vector3f light_cord = samp_pos_light.coords;
// 	Vector3f light_nor = samp_pos_light.normal;
// 	Vector3f light_emit = samp_pos_light.emit;

// 	// 障碍测试, p 点与 采样点 之间是否有遮挡
// 	Vector3f p2light = light_cord - p_cord; // Vector3f p -> light
// 	float distance2_p2light = p2light.norm(); // 采样点与P 点距离平方

// 	Vector3f p2light_n= normalize(p2light);// Vector3f p -> light, 归一化
// 	Ray ray_test(p_cord, normalize(p2light)); // ray p -> light
// 	Intersection inter_test = intersect(ray_test); // 从 object -> object
// 	float distance2_p2inter = (inter_test.coords - p_cord).norm(); // 采样点与P 点距离平方
// 	if(std::fabs(distance2_p2inter - distance2_p2light) < 0.1) 
// 		L_dir = light_emit *p_m->eval(0, -p2light_n, p_nor) *
// 			 dotProduct(-p2light_n, p_nor) * dotProduct(p2light_n, light_nor) / distance2_p2light / pdf_light;

// 	/*** other reflectors ***/
// 	// if(get_random_float() > RussianRoulette) 认为贡献为0, 直接忽略
// 	if(get_random_float() < RussianRoulette)
// 	{
// 		// 从 p 点随机采样wo
// 		Vector3f p_wo = normalize(p_m->sample(Vector3f(0), p_nor));
// 		Ray ray_p_wo(p_cord, p_wo);
// 		Intersection interP2Q = intersect(ray_p_wo); // 从 object p -> object q
// 		// 获取相交object q 的交点的各种属性信息
// 		Material *q_m = interP2Q.m;   // q点材质
// 		Vector3f q_nor = interP2Q.normal;  // q点法线
// 		Vector3f q_cord = interP2Q.coords; // q点坐标
		
// 		// castray 中的hit nothing, 表示直接逃逸了, 可认为hit background, 因此 return background
// 		// 1 如果这里 hit nothing, 则说明这条路径贡献为0, return 0, 所以这里可以忽略

// 		// 2 如果 hit a light, 在 direct light 中已经考虑了, 所以这里忽略
		
// 		// 2.1 hit a non-emitting object
// 		if (interP2Q.happened && !q_m->hasEmission())
// 		{
// 			// 在object q 上随机采样 1 条 wo
// 			Vector3f q_wo = normalize(q_m->sample(Vector3f(0), q_nor));
// 			// 继续递归
// 			float pdf_q = q_m->pdf(0, q_wo, q_nor);
// 			if (pdf_q > EPSILON) // pdf_object != 0
// 				L_indir = MyShade(interP2Q, q_wo) * q_m->eval(0, q_wo, q_nor) * (dotProduct(q_wo, q_nor)) / pdf_q / RussianRoulette;
// 		}
// 	}
// 	return L_dir + L_indir;
// }

// //Implementation of Path Tracing
// Vector3f Scene::castRay(const Ray &ray, int depth) const
// {
//     // TO DO Implement Path Tracing Algorithm here

// 	// 求交
// 	// 依次调用 Scene::intersect, BVHAccel::Intersect, BVHAccel::getIntersection, Bounds3::IntersectP, Triangle::getIntersection
// 	Intersection interP2O = intersect(ray); // 从 eye -> Scene 中的 Object
	
// 	// 获取相交object 的各种属性信息
// 	Material* m = interP2O.m; // 交点材质
// 	Vector3f nor = interP2O.normal; // 交点法线
// 	Vector3f cord =  interP2O.coords; // 交点坐标

// 	if(!interP2O.happened) return this->backgroundColor; // 如果 hit noting, 表示 hit background only, 所以 return backgroundColor
	
// 	// 到这里一定是 hit an object

// 	// 如果 hit a light
// 	//if(m->hasEmission()) {return m->m_emission;}

// 	// 如果材质不发光, hit a non-emitted object
// 	// 随机采样 1 条 wo (在path 的角度是wo, 但在光线的角度是wi)
// 	// 第一个参数没用到, 随便给一个, nor 是交点法线; wo 是由内向外的
// 	Vector3f wo = normalize(m->sample(Vector3f(0), nor));
// 	return MyShade(interP2O, wo);
// }

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
	        // Vector3f p_deviation = (dotProduct(ray.direction,  p_nor) < 0) ?
            //     p_cord +  p_nor * EPSILON :
            //     p_cord -  p_nor * EPSILON ;
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
			float pdf_p = q_m->pdf(w_any, p_wo, p_nor); // p 点 pdf
			if (pdf_p > EPSILON) // pdf_p != 0
				L_indir = castRay(ray_p_wo, depth) * p_m->eval(w_any, p_wo, p_nor) * (dotProduct(p_wo, p_nor)) / pdf_p / RussianRoulette;
		}
	}
	return p_m->getEmission() + L_dir + L_indir;
}

// // Implementation of Path Tracing
// Vector3f Scene::castRay(const Ray& ray, int depth) const
// {
//     // TO DO Implement Path Tracing Algorithm here
//     Vector3f hitColor = this->backgroundColor;
//     Intersection shade_point_inter = Scene::intersect(ray);
//     if (shade_point_inter.happened)
//     {

//         Vector3f p = shade_point_inter.coords;
//         Vector3f wo = ray.direction;
//         Vector3f N = shade_point_inter.normal;
//         Vector3f L_dir(0), L_indir(0);

//        //sampleLight(inter,pdf_light)
//         Intersection light_point_inter;
//         float pdf_light;
//         sampleLight(light_point_inter, pdf_light);
//         //Get x,ws,NN,emit from inter
//         Vector3f x = light_point_inter.coords;
//         Vector3f ws = normalize(x-p); // light to p
//         Vector3f NN = light_point_inter.normal;
//         Vector3f emit = light_point_inter.emit;
//         float distance_pTox = (x - p).norm();
//         //Shoot a ray from p to x
//         Vector3f p_deviation = (dotProduct(ray.direction, N) < 0) ?
//                 p + N * EPSILON :
//                 p - N * EPSILON ;

//         Ray ray_pTox(p_deviation, ws);
//         //If the ray is not blocked in the middleff
//         Intersection blocked_point_inter = Scene::intersect(ray_pTox);
//         if (abs(distance_pTox - blocked_point_inter.distance < 0.01 ))
//         {
//             L_dir = emit * shade_point_inter.m->eval(wo, ws, N) * dotProduct(ws, N) * dotProduct(-ws, NN) / (distance_pTox * distance_pTox * pdf_light);
//         }
//         //Test Russian Roulette with probability RussianRouolette
//         float ksi = get_random_float();
//         if (ksi < RussianRoulette)
//         {
//             //wi=sample(wo,N)
//             Vector3f wi = normalize(shade_point_inter.m->sample(wo, N));
//             //Trace a ray r(p,wi)
//             Ray ray_pTowi(p_deviation, wi);
//             //If ray r hit a non-emitting object at q
//             Intersection bounce_point_inter = Scene::intersect(ray_pTowi);
//             if (bounce_point_inter.happened && !bounce_point_inter.m->hasEmission())
//             {
//                 float pdf = shade_point_inter.m->pdf(wo, wi, N);
//                 if(pdf> EPSILON)
//                     L_indir = castRay(ray_pTowi, depth + 1) * shade_point_inter.m->eval(wo, wi, N) * dotProduct(wi, N) / (pdf *RussianRoulette);
//             }
//         }
//         hitColor = shade_point_inter.m->getEmission() + L_dir + L_indir;
//     }
//     return hitColor;
// }

// static const Vector3f kZeroVector(0.0f);
// Vector3f Scene::castRay(const Ray& ray, int depth) const
// {
//     if (depth > maxDepth) // 检查递归深度
//         return Vector3f(0.0f);

//     Intersection interE2P = intersect(ray);

//     if (!interE2P.happened)
//         return this->backgroundColor;

//     Vector3f p_cord = interE2P.coords;
//     Material* p_m = interE2P.m;
//     Vector3f p_nor = interE2P.normal;

//     Vector3f L_dir = 0;
//     Vector3f L_indir = 0;

//     /*** direct light ***/
//     float pdf_light; Intersection samp_pos_light;
//     sampleLight(samp_pos_light, pdf_light);
//     Vector3f light_cord = samp_pos_light.coords;
//     Vector3f light_nor = samp_pos_light.normal;
//     Vector3f light_emit = samp_pos_light.emit;

//     Vector3f p2light = light_cord - p_cord;
//     float distance2_p2light = p2light.norm();
//     Vector3f p2light_n = normalize(p2light);

//     Ray ray_test(p_cord + p_nor * EPSILON, p2light_n); // 正确的偏移
//     Intersection inter_test = intersect(ray_test);

//     if (std::abs((inter_test.coords - p_cord).norm() - distance2_p2light) < 0.01)
//     {
//         L_dir = light_emit * p_m->eval(kZeroVector, p2light_n, p_nor) * // 使用 kZeroVector
//                 dotProduct(p2light_n, p_nor) * dotProduct(-p2light_n, light_nor) / distance2_p2light / pdf_light;
//     }

//     /*** other reflectors ***/
//     if (get_random_float() < RussianRoulette)
//     {
//         Vector3f p_wo = normalize(p_m->sample(w_any, p_nor));
//         Ray ray_p_wo(p_cord + p_nor * EPSILON, p_wo); // 正确的偏移
//         Intersection interP2Q = intersect(ray_p_wo);

//         if (interP2Q.happened && !interP2Q.m->hasEmission())
//         {
//             float pdf_p = p_m->pdf(kZeroVector, p_wo, p_nor); // 使用 kZeroVector
//             if (pdf_p > EPSILON)
//                 L_indir = castRay(ray_p_wo, depth + 1) * p_m->eval(kZeroVector, p_wo, p_nor) * (dotProduct(p_wo, p_nor)) / pdf_p / RussianRoulette; // 使用 kZeroVector
//         }
//     }
//     return p_m->getEmission() + L_dir + L_indir;
// }