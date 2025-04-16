# 渲染管线实践与优化

## Rasterization

### 1. 构建 MVP 变换矩阵

#### 核心功能

构建 `Model Transform Matrix` 和 `Projection Transform Matrix`.  

#### 技术实现

```  c++
get_model_matrix(float rotation_angle)
```

- 逐个元素地构建模型变换矩阵并返回该矩阵。  

`````c++
get_projection_matrix(float eye_fov, float aspect_ratio, float zNear, float zFar) 
`````

- 使用给定的参数逐个元素地构建透视投影矩阵并返回该矩阵。

#### 成果展示

<figure style="text-align: center;">
  <img src=".\README_Images\01_Result.png" width="80%" />
  <figcaption></figcaption>
</figure>

### 2. 实现光栅化基础功能

#### 核心功能

- 实现深度测试, 缓冲区更新。

#### 技术实现

```c++
void rst::rasterizer::rasterize_triangle(const Triangle& t)
```

- 创建 `bounding box`, 加速边界判断。

- 遍历 `bounding box` 内的所有像素, 使用向量积检查点是否在三角形内。

- 如果在内部，则将其位置处的插值深度值 (interpolated depth value) 与深度
  缓冲区 (depth buffer) 中的相应值进行比较。

- 如果当前点更靠近相机，则设置更新颜色缓冲区及深度缓冲区 (depth buffer)。

- 优化: 应用 `SSAA`模糊化图像边界锯齿。

#### 成果展示

<figure style="text-align: center;">
  <img src=".\README_Images\02_Result.png" width="80%" />
  <figcaption>2-2 Default Result</figcaption>
</figure>

<figure style="text-align: center;">
  <img src=".\README_Images\02_ResultSSAA.png" width="80%" />
  <figcaption>2-2 Result With SSAA</figcaption>
</figure>


### 3. 完善光照模型, 着色模型, 纹理映射

#### 核心功能

- 光照模型: `Blinn-Phong`.
- 着色模型: `Phong.`
- 表面细节增强: `Texture Mapping, Normal Mapping, Bump Mapping, Displacement Mapping.`
- 优化: 应用双线性插值纹理过滤。

#### 技术实现

```c++
void rst::rasterizer::rasterize_triangle(const Triangle& t, 
																	const std::array<Eigen::Vector3f, 3>& view_pos)
```

- 在原有实现基础上增加透视插值矫正的功能,  完成对顶点位置, 法线, 颜色, 纹理坐标的插值计算。

```c++
Eigen::Vector3f phong_fragment_shader(const fragment_shader_payload& payload)
```

- 依据重心插值计算得到的点的颜色(Albedo), 代入 Bliin-Phong 光照模型计算渲染结果。

```c++
Eigen::Vector3f texture_fragment_shader(const fragment_shader_payload& payload)
```

- 从`payload`获取重心插值后的纹理坐标, 使用纹理坐标采样颜色纹理得到的颜色作为像素的颜色。
- 代入` Bliin-Phong` 光照模型计算渲染结果。

```c++
Eigen::Vector3f normal_fragment_shader(const fragment_shader_payload& payload)
```

- 一个简化版本, 仅实现了核心技术要点。
- 没有从法线贴图采样得到目标法线, 而是直接从`payload`获取重心插值后的法线, 归一化并映射到[0,1], 然后直接作为像素的颜色。

```c++
Eigen::Vector3f bump_fragment_shader(const fragment_shader_payload& payload)
```

- 计算高度纹理采样值在`UV` 方向上的变化率。
- 基于`UV`方向的切向量, 计算切线空间中的`normal`, 然后使用`TBN`转换到观察空间(在观察空间完成渲染)。
- 做了一些简化, 没有继续基于 `Bliin-Phong `光照模型进行渲染, 而是直接输出了法线可视化的结果。

```c++
Eigen::Vector3f displacement_fragment_shader(const fragment_shader_payload& payload)
```

- 与前面几种方式不同, 这里以高度纹理采样值作为像素点坐标的增量, 直接改变点的位置

#### 成果展示

<figure style="text-align: center;">
  <img src=".\README_Images\03_phong.png" width="80%" />
  <figcaption>3-1 Bliin-Phong Based on Vertex Color</figcaption>
</figure>

<figure style="text-align: center;">
  <img src=".\README_Images\03_texture.png" width="80%" />
  <figcaption>3-2 Bliin-Phong Based on texture Mapping</figcaption>
</figure>

<figure style="text-align: center;">
  <img src=".\README_Images\03_normal.png" width="80%" />
  <figcaption>3-3 Normal Mapping(Normal Visualization)</figcaption>
</figure>

<figure style="text-align: center;">
  <img src=".\README_Images\03_bump.png" width="80%" />
  <figcaption>3-3 Bump Mapping(Normal Visualization)</figcaption>
</figure>

<figure style="text-align: center;">
  <img src=".\README_Images\03_displacement.png" width="80%" />
  <figcaption>3-4 Bliin-Phong with Displacement Mapping</figcaption>
</figure>

## Geometry

### Bezier 曲线

#### 核心功能

- 基于定义生成`Bezier曲线`。
- `递推(de Casteljau)`算法生成`Bezier`曲线。

#### 技术实现

**这里代码比较简单, 直接引用代码**

```c++
void naive_bezier(const std::vector<cv::Point2f> &points, cv::Mat &window) 
{
    auto &p_0 = points[0];
    auto &p_1 = points[1];
    auto &p_2 = points[2];
    auto &p_3 = points[3];

    for (double t = 0.0; t <= 1.0; t += 0.001) 
    {
        auto point = std::pow(1 - t, 3) * p_0 + 3 * t * std::pow(1 - t, 2) * p_1 					+ 
    		3 * std::pow(t, 2) * (1 - t) * p_2 + std::pow(t, 3) * p_3; // 伯恩斯坦多项式作为权重
        window.at<cv::Vec3b>(point.y, point.x)[2] = 255;
    }
}

// de Casteljau 算法 - 递归
cv::Point2f recursive_bezier(const std::vector<cv::Point2f> &control_points, float t)
{
    if (control_points.size() == 1)
    	return control_points[0];

    std::vector<cv::Point2f> new_points;
    for (size_t i = 0; i < control_points.size() - 1; ++i) {
        cv::Point2f point = (1 - t) * control_points[i] + t * control_points[i + 1];
        new_points.push_back(point);
    }
    return recursive_bezier(new_points, t);
}

void bezier(const std::vector<cv::Point2f> &control_points, cv::Mat &window) 
{
	for (double t = 0.0; t <= 1.0; t += 0.001) 
    {
        cv::Point2f point =  recursive_bezier(control_points, t);
        window.at<cv::Vec3b>(point.y, point.x)[1] = 255;
    }
}
```



#### 成果展示

<figure style="text-align: center;">
  <img src=".\README_Images\04_Bezier Curve.png" width="80%" />
  <figcaption>4 Bezier Curve</figcaption>
</figure>

## Ray Tracing

### 基础功能实现

#### 核心功能

- 查询观察位置与像素点构成的`ray`是否与场景物体相交, 如果相交, 求解最近交点。
- 以交点为起点递归计算像素点应该显示的颜色。

#### 技术实现

```c++
std::optional<hit_payload> trace(const Vector3f &orig, const Vector3f &dir,
        									const std::vector<std::unique_ptr<Object> > &objects)
```

- 遍历场景中所有物体, 求解以`orig`为起点, ` dir `方向的`ray`与物体是否相交, 若相交, 求解最近交点位置, 并存储交点信息。
- 优化:  `Moller-Trumbore `算法加速求交。

```c++
Vector3f castRay(const Vector3f &orig, const Vector3f &dir,
                 const Scene& scene, int depth)
```

- 将以`orig`为起点, ` dir `方向的`ray`为参数调用 `trace函数`求解交点。
- 若与任何物体均无交点, 直接返回背景色。
- 若有交点, 根据`hitPoint`材质信息决定应该如何递归。
  - 反射类材质
    - 向反射方向投射新的`ray`继续递归调用`castRay函数。`
    - 下一层递归返回的`hitColor`即可作为当前`hitPoint`的颜色。
  - 反射+折射类材质
    - 同时向反射和折射方向投射新的`ray`并同时递归调用`castRay函数`。
    - 将下一层递归返回的`hitColor`代入菲涅尔方程计算当前hitPoint的颜色, 即当前`hitColor`.
  - `diffuse`类材质
    - 对`hitPoint`进行 `shadow testing`, 然后对`hitPoint`进行渲染, 计算`hitColor`。
  - 若递归超过最大层数(即最终没有与任何`diffuse`类材质相交, 返回黑色)。

#### 成果展示

<figure style="text-align: center;">
  <img src=".\README_Images\05_RayTracing.png" width="80%" />
  <figcaption>5 Ray Tracing</figcaption>
</figure>


### AABB 和 BVH 加速求交

#### 核心功能

- 构建`BVH`, 并为每个BVH节点构建`包围盒`.
- 先判断与包围盒的相交, 再与物体求交
- 优化: 将`包围盒`视为边界无限大的`AABB`, 加速`包围盒`求交。

#### 技术实现

```c++
BVHBuildNode* BVHAccel::recursiveBuild(std::vector<Object*> objects)
```

- 递归分隔`object`, 构建`BVH`, 仅在叶子节点中存储`object`。
  - 若分隔后仅`1`个`object`, 则为叶子节点, 为该`object`构建包围盒。
  - 若分隔后仅`2`个`objects`, 分别递归, 为每个`object`构建独立的包围盒。
  - 若分割后的`object > 2`, 则继续分隔为两个更小的部分, 分别递归继续分隔。
    - 创建能够包围`objects` 的最小包围盒, 以包围盒的最长轴为依据划分。
      - 假设最长轴为x轴, 则对`objects 以x坐标为准排序`, 将排序后的`objects`分隔为两部分。
      - `yz `轴同理。
  - 递归返回, 以子节点为基准创建父节点的包围盒。

```c++
Intersection BVHAccel::getIntersection(BVHBuildNode* node, const Ray& ray) const
```

- 对于叶子节点, 先与`包围盒`求交, 若有交, 再进一步与叶子节点中的`object`求交。
- 对于非叶子节点, 先与当前节点的`包围盒`求交, 若有交, 再进一步与子节点`包围盒`递归求交。

```c++
inline bool Bounds3::IntersectP(const Ray& ray) const // Devin: 调整了代码框架
{
	double tEnter = -std::numeric_limits<double>::infinity(); // < 0
	double tExit = std::numeric_limits<double>::infinity();// > 0
	for(int i = 0; i < 3; ++i)
	{
		double t1 = (pMin[i] - ray.origin[i]) * ray.direction_inv[i];
		double t2 = (pMax[i] - ray.origin[i]) * ray.direction_inv[i];
		
		if(ray.dirIsNeg[i] == 0) std::swap(t1, t2); // 确保 t1 < t2

		tEnter = std::max(t1, tEnter); // 第一次一定会更新为 t1
		tExit = std::min(t2, tExit); // 第一次一定会更新为 t2
		if(tEnter > tExit) return false;
	}
	return tExit > 0 && tEnter < tExit;
}
```

- 找到最后一个进入盒子的分量, 才表示`ray`真的进入了盒子, 因为不存在有的分量进入, 而有的分量没进入的情况。
- `exit` 同理。
- `exit>0`, 排除ray起点在盒子前面, 即ray反向延长线与盒子相交的情况。
- `exit >= entry`, 这是显然的。

#### 成果展示

<figure style="text-align: center;">
  <img src=".\README_Images\06_RayTracing.png" width="80%" />
  <figcaption>6 Ray Tracing with BVH and AABB</figcaption>
</figure>


## Path Tracing

### 核心功能

- Path Tracing 计算全局光照。
- 蒙特卡洛积分估算渲染方程积分。
- 俄罗斯轮盘赌控制递归停止。
- 优化: 多线程执行 `Scene::castRay 任务`。

### 技术实现

```c++
Vector3f Scene::castRay(const Ray &ray, int depth) const
```

- 从每个`pixel trace n 个 ray,` 计算`n 个 ray` 的平均值。
- hit nothing, return background.
- 计算直接光照。
  - 由于从`hitPoint `采样`ray wo`有概率`hit nothing`导致浪费, 因而这里直接从光源随机采样, 判断光源上的dA与`hitPoint`之间是否有障碍物, 如果无障碍物, 则计算Li(光源)照射下,` hit Point的Lo`
- 计算间接光照。
  - 引入俄罗斯轮盘赌机制`(true 的概率为P)`控制hitPoint是否trace新的ray, 进而`控制递归停止`。
  - 若trace新的ray, 则使用`蒙特卡洛采样ray 的方向`, 继续递归。
  - 计算出的`Lo / P`(归一化, 确保能量守恒)。
- 返回 `lightRadiance(若hitPoint为发光材质, 则不为0) + LDirec + LIndirect`.

```c++
void Renderer::Render(const Scene& scene)
```

- 创建多个线程执行 `Scene::castRay 任务`, 每个任务负责计算`framebuffer的一个特定区域的像素trace 的ray`.

### 成果展示

<figure style="text-align: center;">
  <img src=".\README_Images\07_PathTracing_16SPP.png" width="80%" />
  <figcaption>7-1 Path Tracing, SPP = 16, Time Taken: 2 min</figcaption>
</figure>

<figure style="text-align: center;">
  <img src=".\README_Images\07_PathTracing_64SPP.png" width="80%" />
  <figcaption>7-2 Path Tracing, SPP = 64, Time Taken: 12 min</figcaption>
</figure>

<figure style="text-align: center;">
  <img src=".\README_Images\07_PathTracing_256SPP.png" width="80%" />
  <figcaption>7-3 Path Tracing, SPP = 256, Time Taken: 54 min</figcaption>
</figure>