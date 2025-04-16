#include <fstream>
#include "Vector.hpp"
#include "Renderer.hpp"
#include "Scene.hpp"
#include <optional>

inline float deg2rad(const float &deg)
{ return deg * M_PI/180.0; }

// Compute reflection direction // 计算反射方向
Vector3f reflect(const Vector3f &I, const Vector3f &N)
{
    return I - 2 * dotProduct(I, N) * N;
}

// [comment]
// Compute refraction direction using Snell's law
//
// We need to handle with care the two possible situations:
//
//    - When the ray is inside the object
//
//    - When the ray is outside.
//
// If the ray is outside, you need to make cosi positive cosi = -N.I // 光线从外部进入, 
//
// If the ray is inside, you need to invert the refractive indices and negate the normal N
// [/comment]
Vector3f refract(const Vector3f &I, const Vector3f &N, const float &ior)
{
	// 余弦函数的值域本身就在 ([-1, 1]) 之间，但由于浮点运算的精度问题，计算结果可能会略微超出这个范围
	// 通过使用 clamp 函数，可以确保 cosi 的值始终在合法范围内，避免后续计算中的潜在错误
    float cosi = clamp(-1, 1, dotProduct(I, N));
    float etai = 1, etat = ior;// etai 是入射介质的折射率，etat 是出射介质的折射率
    Vector3f n = N;

	// 为了符合 Snell's law, 要对cosi 取正
	// cosi < 0, 说明光线从外部进入内部
	// // 此时法向量的方向与折射线的方向是相反的, 因此取负值
	// // 同时交换入射折射率和出射折射率
    if (cosi < 0) { cosi = -cosi; } else { std::swap(etai, etat); n= -N; }
    float eta = etai / etat; // 折射率比
	// 1 - cosi * cosi = (sini)^2
	// eta * eta * (sini)^2 = (sint)^2, 出射角正弦的平方
	// 1 - (sint)^2 = (cost)^2, 即 k 是出射角余弦的平方
	// 当光线从高密度介质(内部)进入低密度介质(外部), 且入射角大于临界角时, 会发生全反射现象, 导致 (sini)^2 > (1 / eta)^2, 导致 k < 0
    float k = 1 - eta * eta * (1 - cosi * cosi);  
    return k < 0 ? 0 : eta * I + (eta * cosi - sqrtf(k)) * n;
}

// [comment]
// Compute Fresnel equation
//
// \param I is the incident view direction
//
// \param N is the normal at the intersection point
//
// \param ior is the material refractive index
// [/comment]
float fresnel(const Vector3f &I, const Vector3f &N, const float &ior)
{
    float cosi = clamp(-1, 1, dotProduct(I, N));// 入射角余弦, 避免浮点数计算误差,因此限制[-1, 1] 的范围
    float etai = 1, etat = ior;
    if (cosi > 0) {  std::swap(etai, etat); } // 说明从高密度介质(外部)进入低密度介质(内部), 交换二者的折射率
    // Compute sini using Snell's law
    float sint = etai / etat * sqrtf(std::max(0.f, 1 - cosi * cosi)); // 折射角正弦
    // Total internal reflection, 
    if (sint >= 1) {
        return 1;// 全反射, 反射比 为1
    }
    else {
        float cost = sqrtf(std::max(0.f, 1 - sint * sint));
        cosi = fabsf(cosi);//为了符合 Fresnel Equation 的要求, consi 取正
		
        float Rs = ((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost)); //如果入射光是s偏振
        float Rp = ((etai * cosi) - (etat * cost)) / ((etai * cosi) + (etat * cost));//如果入射光是p偏振

		// 上面两个式子似乎有问题,少一个平方, 先放到这里
		// float Rs = std::pow(((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost)), 2); //如果入射光是s偏振
        // float Rp = std::pow(((etai * cosi) - (etat * cost)) / ((etai * cosi) + (etat * cost)), 2);//如果入射光是p偏振
        return (Rs * Rs + Rp * Rp) / 2; // 这里按非偏振光计算,反射比是两者的算数平均值
    }
    // As a consequence of the conservation of energy, transmittance is given by:
    // kt = 1 - kr;// 折射比 = 1 - 反射比
}

// [comment]
// Returns true if the ray intersects an object, false otherwise.
//
// \param orig is the ray origin
// \param dir is the ray direction
// \param objects is the list of objects the scene contains
// \param[out] tNear contains the distance to the cloesest intersected object.
// \param[out] index stores the index of the intersect triangle if the interesected object is a mesh.
// \param[out] uv stores the u and v barycentric coordinates of the intersected point
// \param[out] *hitObject stores the pointer to the intersected object (used to retrieve material information, etc.)
// \param isShadowRay is it a shadow ray. We can return from the function sooner as soon as we have found a hit.
// [/comment]
std::optional<hit_payload> trace(
        const Vector3f &orig, const Vector3f &dir,
        const std::vector<std::unique_ptr<Object> > &objects)
{
    float tNear = kInfinity;
    std::optional<hit_payload> payload;
    for (const auto & object : objects) // 遍历场景中的物体
    {
        float tNearK = kInfinity;
        uint32_t indexK;
        Vector2f uvK;
        if (object->intersect(orig, dir, tNearK, indexK, uvK) && tNearK < tNear) // 
        {
            payload.emplace(); // 构造一个空的 hit_payload 对象
            payload->hit_obj = object.get();//从 unique_ptr 中获取 object 的指针
            payload->tNear = tNearK; // 存储与这条光线相交的最近的交点的 z 值
            payload->index = indexK;
            payload->uv = uvK;
            tNear = tNearK;
        }
    }

    return payload;
}

// [comment]
// Implementation of the Whitted-style light transport algorithm (E [S*] (D|G) L)
//
// 该函数计算由位置和方向定义的光线在交点处的颜色, 该函数是递归的
// This function is the function that compute the color at the intersection point
// of a ray defined by a position and a direction. Note that thus(this?) function is recursive (it calls itself).
//
// 当材质是反射或反射 + 折射, 需要计算发射及折射的方向, 然后递归调用 castRay函数将新的光线投射到场景中
// If the material of the intersected object is either reflective or reflective and refractive,
// then we compute the reflection/refraction direction and cast two new rays into the scene
// by calling the castRay() function recursively.
//
// 如果表面是透明的, 使用 Fresnel Equations 的结果, 混合折射与反射的颜色
// When the surface is transparent, we mix the reflection and refraction color using the result of the fresnel equations. 
// (it computes the amount of reflection and refraction depending on the surface normal,
// incident view direction and surface refractive index).
//
// 如果是漫反射或抛光材质, 则使用 Phong 光照模型计算交点处的颜色
// If the surface is diffuse/glossy we use the Phong illumation model to compute the color at the intersection point.
// [/comment]
Vector3f castRay(
        const Vector3f &orig, const Vector3f &dir, const Scene& scene,
        int depth)
{
    if (depth > scene.maxDepth) {
        return Vector3f(0.0,0.0,0.0);
    } // 递归超过最大深度, 返回黑色

    Vector3f hitColor = scene.backgroundColor; // 初始颜色为背景色
	// 这里是判断从观察位置出发的"光线" 是否与场景中的物体相交
	// C++17语法, 声明一个optional对象, 并通过 trace 获得返回值, 如果返回值存在, 则执行if语句
    if (auto payload = trace(orig, dir, scene.get_objects()); payload) 
    {
        Vector3f hitPoint = orig + dir * payload->tNear; // o + t * d, 观察位置起点 -> 交点
        Vector3f N; // normal
        Vector2f st; // st coordinates
		// hitPoint, 交点; dir, 光线方向; index, 交点所属三角形的索引; uv, 交点的重心坐标; N, 法向量; st, 纹理坐标(输出型参数)
		// 多态调用, 获取 Object 表面属性(好像只获取了纹理坐标?)
        payload->hit_obj->getSurfaceProperties(hitPoint, dir, payload->index, payload->uv, N, st); 
        switch (payload->hit_obj->materialType) {
            case REFLECTION_AND_REFRACTION://反射 + 折射 类材质
            {
                Vector3f reflectionDirection = normalize(reflect(dir, N));
                Vector3f refractionDirection = normalize(refract(dir, N, payload->hit_obj->ior));
                Vector3f reflectionRayOrig = (dotProduct(reflectionDirection, N) < 0) ?
                                             hitPoint - N * scene.epsilon :
                                             hitPoint + N * scene.epsilon;
				// < 0,说明光线从内部射向外部, 给交点一个非常非常小的向内的偏移量
				// > 0,说明光线从外部部射向内部, 给交点一个非常非常小的外的偏移量
				// 很小的偏移量可以区分内外,避免光线与自己相交,同时也不至于影响渲染效果
                Vector3f refractionRayOrig = (dotProduct(refractionDirection, N) < 0) ?
                                             hitPoint - N * scene.epsilon : 
                                             hitPoint + N * scene.epsilon; 
                Vector3f reflectionColor = castRay(reflectionRayOrig, reflectionDirection, scene, depth + 1); // 投射新的反射光线
                Vector3f refractionColor = castRay(refractionRayOrig, refractionDirection, scene, depth + 1); // 投射新的折射光线
                float kr = fresnel(dir, N, payload->hit_obj->ior); // 计算反射比
				// 加权平均, 折射比 与 反射比 为加权系数
				// 本质而言, 这里以经过发射折射效果后的形成的新的颜色继续去照射其它物体
				// 最终递归停止的物体的颜色就是?
                hitColor = reflectionColor * kr + refractionColor * (1 - kr); 
                break;
            }
            case REFLECTION://仅反射类材质(本练习没有这样的材质)
            {
                float kr = fresnel(dir, N, payload->hit_obj->ior);
				// 只要法线, 基础光线, 折射光线单位化了, 这里就没有必要单位化, 但是我觉得单位化后, 代码更健壮
				// Vector3f reflectionDirection = normalize(reflect(dir, N)); 
                Vector3f reflectionDirection = reflect(dir, N);
                Vector3f reflectionRayOrig = (dotProduct(reflectionDirection, N) < 0) ? // 同样很小的偏移量
                                             hitPoint + N * scene.epsilon :
                                             hitPoint - N * scene.epsilon;
                hitColor = castRay(reflectionRayOrig, reflectionDirection, scene, depth + 1) * kr; // 反射比kr: 反射能量 / 入射能量
                break;
            }
			// DEFFUSE_AND_GLOSSY
			// 没有参与递归, 是不是说其它的折射反射对它都没有影响?
            default: 
            {
                // [comment]
				// 这里考虑漫反射和高光项
                // We use the Phong illumation model int the default case. The phong model
                // is composed of a diffuse and a specular reflection component.
                // [/comment]
                Vector3f lightAmt = 0, specularColor = 0;
                Vector3f shadowPointOrig = (dotProduct(dir, N) < 0) ? // 避免光线自交,详细原因见上
                                           hitPoint + N * scene.epsilon :
                                           hitPoint - N * scene.epsilon;
                // [comment]
                // Loop over all lights in the scene and sum their contribution up
                // We also apply the lambert cosine law
                // [/comment]
                for (auto& light : scene.get_lights()) { // 本例提供了两个点光源
                    Vector3f lightDir = light->position - hitPoint; // 光线方向hitPoint -> light position
                    // square of the distance between hitPoint and the light
                    float lightDistance2 = dotProduct(lightDir, lightDir); // 光源到 交点的距离的平方
                    lightDir = normalize(lightDir);//单位化光线方向向量
                    float LdotN = std::max(0.f, dotProduct(lightDir, N)); // 光线与法向量夹角的余弦值
                    // is the point in shadow, and is the nearest occluding object closer to the object than the light itself?
					// 这里是求解 hitPoint 指向光源方向的一条线与路径中所有 object 相交的最近的交点, 存储在 shadow_res 中
                    auto shadow_res = trace(shadowPointOrig, lightDir, scene.get_objects()); 
					// 如果 hadow_res 为真, 说明有交点, 进一步判断相交物体是否在光源与hitPoint之间, 即是否会阻挡光源照射到 hitPoint
					// shadow_res->tNear 是光源与 hitPoint 这条线的路径上,所有相交物体的最近的交点的 z 值
					// 如果  lightDistance > tNear, 说明光源与hitPoint 之间有其它物体, 则 thiPoint 应该在阴影中
                    bool inShadow = shadow_res && (shadow_res->tNear * shadow_res->tNear < lightDistance2);

					// 漫反射光强
					// 如果在阴影中, 不计算光源的贡献, 直接认为光强为0
					// 不在阴影中, 要考虑 lambert cosine law, I * cosα
                    lightAmt += inShadow ? 0 : light->intensity * LdotN;
					// 因为 lightDir 是从 hitPoint -> light position 指向光源的,实际应该从光源指向 hitPoint 才是合理的, 所以这里取反
                    Vector3f reflectionDirection = reflect(-lightDir, N);
					
					//高光项
					//这里计算的是使用了反射方向与观察方向夹角的余弦值,并没有使用中程向量
                    specularColor += powf(std::max(0.f, -dotProduct(reflectionDirection, dir)),
                        payload->hit_obj->specularExponent) * light->intensity; // cosα^m * I
                }
				// evalDiffuseColor(st) 是多态调用: 对球, 直接返回默认颜色,纹理项无效; 对三角形会计算纹理坐标
                hitColor = lightAmt * payload->hit_obj->evalDiffuseColor(st) * payload->hit_obj->Kd + specularColor * payload->hit_obj->Ks;
                break;
            }
        }
    }
	// 从 default: 中的 hitColor 层层返回
	// default: hitColor 会直接返回给eye(当hitPoint 对应的ray 直接从由 eye cast)
	// default: hitColor 会直接返回给 reflection 或 refraction 计算折射或反射的颜色
	// 然后反折射计算出的 hitColor 会继续返回作为上一层的 折反射颜色计算, 直到最后返回到eye
    return hitColor;
}

// [comment]
// 遍历场景中的像素, 生成基础光线, 投射到场景中
// The main render function. This where we iterate over all pixels in the image, generate
// primary rays and cast these rays into the scene. The content of the framebuffer is
// saved to a file.
// [/comment]
void Renderer::Render(const Scene& scene)
{
    std::vector<Vector3f> framebuffer(scene.width * scene.height); // 创建帧缓存对象

    float scale = std::tan(deg2rad(scene.fov * 0.5f));
    float imageAspectRatio = scene.width / (float)scene.height; //宽高比

    // Use this variable as the eye position to start your rays.
    Vector3f eye_pos(0);
    int m = 0;
    for (int j = 0; j < scene.height; ++j)
    {
        for (int i = 0; i < scene.width; ++i)
        {
            // generate primary ray direction
            float x = (2.f * (i + 0.5f) / (float)(scene.width) - 1.f) * scale * imageAspectRatio;
            float y = (1.f - 2.f * (j + 0.5f) / (float)(scene.height)) * scale;
            // TODO: Find the x and y positions of the current pixel to get the direction
            // vector that passes through it.
            // Also, don't forget to multiply both of them with the variable *scale*, and
            // x (horizontal) variable with the *imageAspectRatio*            

            Vector3f dir = Vector3f(x, y, -1); // Don't forget to normalize this direction!
			dir = normalize(dir); // 单位化
            framebuffer[m++] = castRay(eye_pos, dir, scene, 0);
        }
        UpdateProgress((j + 1) / (float)scene.height);
    }

    // save framebuffer to file
    FILE* fp = fopen("binary.ppm", "wb");
    (void)fprintf(fp, "P6\n%d %d\n255\n", scene.width, scene.height);
    for (auto i = 0; i < scene.height * scene.width; ++i) {
        static unsigned char color[3];
        color[0] = (char)(255 * clamp(0, 1, framebuffer[i].x));
        color[1] = (char)(255 * clamp(0, 1, framebuffer[i].y));
        color[2] = (char)(255 * clamp(0, 1, framebuffer[i].z));
        fwrite(color, 1, 3, fp);
    }
    fclose(fp);
}
