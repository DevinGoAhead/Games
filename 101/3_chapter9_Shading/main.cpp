#include <iostream>
#include <opencv2/opencv.hpp>

#include "global.hpp"
#include "rasterizer.hpp"
#include "Triangle.hpp"
#include "Shader.hpp"
#include "Texture.hpp"
#include "OBJ_Loader.h"

float AngleToRadian(float angle)
{
	return (angle / 180) * MY_PI;
}

float GetHeight(const fragment_shader_payload& payload, float u, float v)
{
	Texture* ptex = payload.texture;

	Eigen:: Vector3f rgb = ptex->getColorBilinear(u, v);
	//rgb.normalize();
	// return std::sqrt(rgb.x() * rgb.x() + rgb.y() * rgb.y() +rgb.z() * rgb.z());
	return rgb.norm();
}

Eigen::Matrix4f get_view_matrix(Eigen::Vector3f eye_pos)
{
    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();

    Eigen::Matrix4f translate;
    translate << 1,0,0,-eye_pos[0],
                 0,1,0,-eye_pos[1],
                 0,0,1,-eye_pos[2],
                 0,0,0,1;

    view = translate*view;

    return view;
}

Eigen::Matrix4f get_model_matrix(float angle)
{
    Eigen::Matrix4f rotation; // 旋转
    angle = angle * MY_PI / 180.f;
    rotation << cos(angle), 0, sin(angle), 0, 
                0, 1, 0, 0,
                -sin(angle), 0, cos(angle), 0,
                0, 0, 0, 1;

    Eigen::Matrix4f scale; // 缩放
    scale << 2.5, 0, 0, 0,
              0, 2.5, 0, 0,
              0, 0, 2.5, 0,
              0, 0, 0, 1;

    Eigen::Matrix4f translate; // 平移, 但其实这里平移的量为0
    translate << 1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1;

    return translate * rotation * scale;
}

Eigen::Matrix4f get_projection_matrix(float eye_fov, float aspect_ratio, float zNear, float zFar)
{
    // TODO: Use the same projection matrix from the previous assignments
	Eigen::Matrix4f projection = Eigen::Matrix4f::Identity();

	////本例给出的 zNear 和 zFar 都是正值(距离), 而推导用的负值(坐标)
	//// 但是神奇的是, 在这里, 去掉取负,反而是正确的, 也不知道是哪里的问题
	// zNear = -zNear;
	// zFar = -zFar;

	// Squish - Perspactive Protection to Orthographic Protection
	Eigen::Matrix4f squish;
 	squish << zNear, 0, 0, 0,
				0, zNear, 0, 0,
				0, 0, zNear + zFar, -(zNear * zFar),
				0, 0, 1, 0;
	
	// Orthographic Projection
	float eyeRadian = AngleToRadian(eye_fov);
	float yTop, yBot, xLeft,xRight;//需要float, 而不是int
	yBot = zNear * tan(eyeRadian / 2);
 	yTop = -yBot;
	xLeft = yBot * aspect_ratio;
	xRight = -xLeft;
	
	Eigen::Matrix4f orthographicProj = Eigen::Matrix4f::Identity();
	Eigen::Matrix4f translation;
	translation << 1, 0, 0, -(xLeft + xRight) / 2,
					0, 1, 0, -(yBot + yTop) / 2,
					0, 0, 1, -(zFar + zNear) / 2,
					0, 0, 0, 1;
	Eigen::Matrix4f scale;
	scale << 2 / (xRight - xLeft), 0, 0, 0,
				0, 2 / (yTop - yBot), 0, 0,
				0, 0, 2 / (zNear - zFar), 0,
				0, 0, 0, 1;
	orthographicProj = scale * translation;
	projection = orthographicProj * squish;
	// std::cout << "Matrix:\n" << projection << std::endl;
    return projection;
}

Eigen::Vector3f vertex_shader(const vertex_shader_payload& payload)
{
    return payload.position;
}

Eigen::Vector3f normal_fragment_shader(const fragment_shader_payload& payload)
{
	// normal.head<3>().normalized(), 将前 3 个分量归一化, 但是 normal 本身就是3个分量, 为什么要强调前3个,不太明白
	// + Eigen::Vector3f(1.0f, 1.0f, 1.0f), 之后再 / 2, 保证其最终的分量均在[0,1] 范围
	// 这里就仅仅是定义一种规则, 使法线映射到颜色, 当然确保法线分量不能出现负值
    Eigen::Vector3f return_color = (payload.normal.head<3>().normalized() + Eigen::Vector3f(1.0f, 1.0f, 1.0f)) / 2.f;
    Eigen::Vector3f result;
    result << return_color.x() * 255, return_color.y() * 255, return_color.z() * 255;
    return result;
}

static Eigen::Vector3f reflect(const Eigen::Vector3f& vec, const Eigen::Vector3f& axis)
{
    auto costheta = vec.dot(axis);
    return (2 * costheta * axis - vec).normalized();
}

struct light
{
    Eigen::Vector3f position;
    Eigen::Vector3f intensity;
};

Eigen::Vector3f texture_fragment_shader(const fragment_shader_payload& payload)
{
    Eigen::Vector3f return_color = {0, 0, 0};
    if (payload.texture)
    {
        // TODO: Get the texture value at the texture coordinates of the current fragment
		return_color = payload.texture->getColor(payload.tex_coords.x(), payload.tex_coords.y());
		//return_color = payload.texture->getColorBilinear(payload.tex_coords.x(), payload.tex_coords.y());
    }
    Eigen::Vector3f texture_color;
	texture_color << return_color.x(), return_color.y(), return_color.z();
    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
	// 这里用纹理替代 Blinn - Phong 模型中 物体本来的颜色
	// payload 中这两个参数都是有的
    Eigen::Vector3f kd = texture_color / 255.f; 
	//std::cout << kd.x() << ", " <<  kd.x() << ", " << kd.z() << std::endl;
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);
	float inte = 500;
    auto l1 = light{{20, 20, 20}, {inte, inte, inte}};
    auto l2 = light{{-20, 20, 0}, {inte, inte, inte}};

    std::vector<light> lights = {l1, l2};
    Eigen::Vector3f amb_light_intensity{10, 10, 10};
    Eigen::Vector3f eye_pos{0, 0, 10};

    float p = 150;

    Eigen::Vector3f color = texture_color;
    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal;

    Eigen::Vector3f result_color = {0, 0, 0};

    Eigen::Vector3f lightAtPoint(0, 0, 0);

    for (auto& light : lights)
    {
        // TODO: For each light source in the code, calculate what the *ambient*, *diffuse*, and *specular* 
        // components are. Then, accumulate that result on the *result_color* object.+
		//if (normal.norm() > 1e-5) normal.normalize();

		normal.normalize(); // 法线向量
		Eigen::Vector3f vLight = light.position - point; // 点 -> 入射光位置 向量
		Eigen::Vector3f vView = eye_pos - point; // 点 -> 视角(摄像机) 向量
		float distance = vLight.norm();

		vLight.normalize();
		vView.normalize();
	
		Eigen::Vector3f besector =  vLight + vView;
		besector.normalize();

		Eigen::Vector3f intensityAtpoint = light.intensity / std::pow(distance, 2); // eye_pos 位置的光强

		// 漫反射项
		Eigen::Vector3f lightDiff = kd.cwiseProduct(intensityAtpoint) * std::max(0.f, normal.dot(vLight));

		// 高光项
		Eigen::Vector3f lightSpec = ks.cwiseProduct(intensityAtpoint) * std::pow(std::max(0.f, normal.dot(besector)), p);

		lightAtPoint = lightAtPoint + lightDiff + lightSpec;
    }

	// 环境光项
	Eigen::Vector3f lightAmb = ka.cwiseProduct(amb_light_intensity);
	
	lightAtPoint += lightAmb;
	result_color = lightAtPoint;
	//result_color = lightAtPoint.cwiseProduct(color / 255.f);
    return result_color * 255.f;
}

Eigen::Vector3f phong_fragment_shader(const fragment_shader_payload& payload)
{
    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005); //环境光系数
    Eigen::Vector3f kd = payload.color; // 漫反射系数
	// std::cout << kd.x() << ", " <<  kd.x() << ", " << kd.z() << std::endl;
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937); // 高光系数

    auto l1 = light{{20, 20, 20}, {500, 500, 500}};// 光源位置与光源强度
    auto l2 = light{{-20, 20, 0}, {500, 500, 500}};
	// auto l1 = light{{20, 20, 20}, {10,10, 10}};
    // auto l2 = light{{-20, 20, 0}, {10, 10,10}};

    std::vector<light> lights = {l1, l2}; // 两个光源
    Eigen::Vector3f amb_light_intensity{10, 10, 10}; // 环境光强度
	// 观察位置坐标
	// 这里应该是有问题的, 因为shading 是在观察空间进行的, model 经过了view 变换, 摄像机也应经过view变换, 即的位置应该为(0, 0, 0)
	// 但是实际测试后, 发现 改为 (0, 0, 0) 也是没有任何变化的, 暂不知原因
    Eigen::Vector3f eye_pos{0, 0, 10}; // 

    float p = 150;

    Eigen::Vector3f color = payload.color; // 像素颜色
    Eigen::Vector3f point = payload.view_pos; // 视空间中 像素坐标
    Eigen::Vector3f normal = payload.normal; // 法线

    Eigen::Vector3f result_color = {0, 0, 0};
	Eigen::Vector3f lightAtPoint(0, 0, 0);

    for (auto& light : lights)
    {
        // TODO: For each light source in the code, calculate what the *ambient*, *diffuse*, and *specular* 
        // components are. Then, accumulate that result on the *result_color* object.+
		normal.normalize(); // 法线向量归一化
		Eigen::Vector3f vLight = light.position - point; // 点 -> 入射光位置 向量
		Eigen::Vector3f vView = eye_pos - point; // 点 -> 视角(摄像机) 向量
		float distance = vLight.norm();

		vLight.normalize();
		vView.normalize();
	
		Eigen::Vector3f besector =  vLight + vView; // 半程向量
		besector.normalize();

		Eigen::Vector3f intensityAtpoint = light.intensity / std::pow(distance, 2); // eye_pos 位置的光强

		// 漫反射项
		Eigen::Vector3f lightDiff = kd.cwiseProduct(intensityAtpoint) * std::max(0.f, normal.dot(vLight));

		// 高光项
		Eigen::Vector3f lightSpec = ks.cwiseProduct(intensityAtpoint) * std::pow(std::max(0.f, normal.dot(besector)), p);

		lightAtPoint = lightDiff + lightSpec;
    }

	// 环境光项
	Eigen::Vector3f lightAmb = ka.cwiseProduct(amb_light_intensity);
	lightAtPoint += lightAmb;
	// 这种方式, 模型本身的颜色仅仅对漫反射光会对有影响, 因为 kd = color;
	// 而高光, 环境光对模型的颜色似乎并没有产生作用
	// 这里应该 直接 = 或是 做 .cwiseProduct 运算? 不确定
	result_color = lightAtPoint; 
	//result_color = lightAtPoint.cwiseProduct(color);
    return result_color * 255.f;
}

Eigen::Vector3f displacement_fragment_shader(const fragment_shader_payload& payload)
{
    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
    Eigen::Vector3f kd = payload.color;
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

    auto l1 = light{{20, 20, 20}, {500, 500, 500}};
    auto l2 = light{{-20, 20, 0}, {500, 500, 500}};

    std::vector<light> lights = {l1, l2};
    Eigen::Vector3f amb_light_intensity{10, 10, 10};
    Eigen::Vector3f eye_pos{0, 0, 10};

    float p = 150;

    Eigen::Vector3f color = payload.color; 
    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal;

    float kh = 0.2, kn = 0.1;
    
    // TODO: Implement displacement mapping here
    // Let n = normal = (x, y, z)
    // Vector t = (x*y/sqrt(x*x+z*z),sqrt(x*x+z*z),z*y/sqrt(x*x+z*z))
    // Vector b = n cross product t
    // Matrix TBN = [t b n]
    // dU = kh * kn * (h(u+1/w,v)-h(u,v))
    // dV = kh * kn * (h(u,v+1/h)-h(u,v))
    // Vector ln = (-dU, -dV, 1)
    // Position p = p + kn * n * h(u,v)
    // Normal n = normalize(TBN * ln)

	Eigen::Vector3f t, b, n;
	n = normal.normalized();
	// t << -n.z(), 0, n.x();
	// t /= std::sqrt(std::pow(n.z(), 2) + std::pow(n.x(), 2) ); // t /= sqrt(x^2 + z^2)
	t << n.x()*n.y()/std::sqrt(n.x()*n.x() + n.z()*n.z()),
		std::sqrt(n.x()*n.x()+n.z()*n.z()),
		n.z()*n.y()/std::sqrt(n.x()*n.x()+n.z()*n.z());
	t.normalize();
	b = n.cross(t).normalized();
	
	Eigen::Matrix3f TBN;
	TBN.col(0) = t;
	TBN.col(1) = b;
	TBN.col(2) = n;

	float u = payload.tex_coords.x();
	float v = payload.tex_coords.y();
	float heightImage = payload.texture->height;
	float widthImage = payload.texture->width;
	float pointHeight = GetHeight(payload, u, v);

	float dU = kh * (GetHeight(payload, std::min(u + 1.f / widthImage, 1.f), v) - pointHeight);
	float dV = kh * (GetHeight(payload, u, std::min(v + 1.f/ heightImage, 1.f)) - pointHeight);

	Eigen::Vector3f nMap(-dU, -dV, 1);
	Eigen::Vector3f nTan =  (TBN * nMap).normalized();

	Eigen::Vector3f displacement = kn * n * pointHeight;
	point += displacement;


   	Eigen::Vector3f result_color(0,0,0);
    normal = nTan;
	Eigen::Vector3f lightAtPoint(0, 0, 0);
	for (auto& light : lights)
    {
        // TODO: For each light source in the code, calculate what the *ambient*, *diffuse*, and *specular* 
        // components are. Then, accumulate that result on the *result_color* object.+
		normal.normalize(); // 法线向量
		Eigen::Vector3f vLight = light.position - point; // 点 -> 入射光位置 向量
		Eigen::Vector3f vView = eye_pos - point; // 点 -> 视角(摄像机) 向量
		float distance = vLight.norm();

		vLight.normalize();
		vView.normalize();
	
		Eigen::Vector3f besector =  vLight + vView;
		besector.normalize();

		Eigen::Vector3f intensityAtpoint = light.intensity / std::pow(distance, 2); // eye_pos 位置的光强

		// 漫反射项
		Eigen::Vector3f lightDiff = kd.cwiseProduct(intensityAtpoint) * std::max(0.f, normal.dot(vLight));

		// 高光项
		Eigen::Vector3f lightSpec = ks.cwiseProduct(intensityAtpoint) * std::pow(std::max(0.f, normal.dot(besector)), p);

		lightAtPoint = lightAtPoint + lightDiff + lightSpec;
    }

	//环境光项
	Eigen::Vector3f lightAmb = ka.cwiseProduct(amb_light_intensity);
	lightAtPoint += lightAmb;
	result_color = lightAtPoint;
	//result_color = lightAtPoint.cwiseProduct(result_color);

    return result_color * 255.f;
}

Eigen::Vector3f bump_fragment_shader(const fragment_shader_payload& payload)
{
    
    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
    Eigen::Vector3f kd = payload.color;
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

    auto l1 = light{{20, 20, 20}, {500, 500, 500}};
    auto l2 = light{{-20, 20, 0}, {500, 500, 500}};

    std::vector<light> lights = {l1, l2};
    Eigen::Vector3f amb_light_intensity{10, 10, 10};
    Eigen::Vector3f eye_pos{0, 0, 10};

    float p = 150;

    Eigen::Vector3f color = payload.color; 
    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal;


    float kh = 0.2, kn = 0.1;

    // TODO: Implement bump mapping here
    // Let n = normal = (x, y, z)
    // Vector t = (x*y/sqrt(x*x+z*z),sqrt(x*x+z*z),z*y/sqrt(x*x+z*z))
    // Vector b = n cross product t
    // Matrix TBN = [t b n]
    // dU = kh * kn * (h(u+1/w,v)-h(u,v))
    // dV = kh * kn * (h(u,v+1/h)-h(u,v))
    // Vector ln = (-dU, -dV, 1)
    // Normal n = normalize(TBN * ln)
	Eigen::Vector3f t, b, n;
	n = normal.normalized();
	// t << -n.z(), 0, n.x();
	// t /= std::sqrt(std::pow(n.z(), 2) + std::pow(n.x(), 2) ); // t /= sqrt(x^2 + z^2)
	
	// 这里应该也是有问题的, 稍加计算可以法线, 除非n.y == 0 或则 n.x == n.z == 0
	// n.z 为主法线方向,  显然,不太可能为0, 因此只能 n.y == 0, 这就相当于默认 法线在 y 方向的分量为0, 显然这是不合理的.
	// 当然, 似乎也没有更好的方法, 因为标准的方法需要使用三角形的三个点,这里仅使用了法向量
	t << n.x()*n.y()/std::sqrt(n.x()*n.x() + n.z()*n.z()),
		std::sqrt(n.x()*n.x()+n.z()*n.z()),
		n.z()*n.y()/std::sqrt(n.x()*n.x()+n.z()*n.z());
	t.normalize();
	b = n.cross(t).normalized();
	
	Eigen::Matrix3f TBN;
	TBN.col(0) = t;
	TBN.col(1) = b;
	TBN.col(2) = n;

	float u = payload.tex_coords.x(); // 纹理坐标
	float v = payload.tex_coords.y(); // 纹理坐标
	float heightImage = payload.texture->height; // uv 图高度
	float widthImage = payload.texture->width;// uv 图宽度
	float pointHeight = GetHeight(payload, u, v); // uv 位置的高度, 其实是rgb 向量的模
	
	// 这里应避免 u + 1.f 越界, v 同理
	float dU = kh * (GetHeight(payload, std::min(u + 1.f / widthImage, 1.f), v) - pointHeight);
	float dV = kh * (GetHeight(payload, u, std::min(v + 1.f/ heightImage, 1.f)) - pointHeight);

	Eigen::Vector3f nMap(-dU, -dV, 1); // 这里推导貌似挺复杂的, 先类比二维理解
	Eigen::Vector3f nTan =  (TBN * nMap); // 从贴图(uv)空间转换到 切线空间
	
	// // 这里是否还需要转换到观察空间?
	// Eigen::Vector4f nView(nTan, 0);
	// // 逆转换到模型空间, 逆转换到世界空间, 转换到观察空间
	// nView = get_view_matrix(eye_pos) * get_model_matrix(140).inverse() * TBN.inverse() * nView; 
	// nView.normalize();// 单位话
	// Eigen::Vector3f result_color = nView.head<3>();

	Eigen::Vector3f result_color = nTan;

    return result_color * 255.f;
}

int main(int argc, const char** argv)
{
    std::vector<Triangle*> TriangleList;

    float angle = 140.0;
    bool command_line = false;

    std::string filename = "output.png"; // 默认输出文件名
    objl::Loader Loader; // 模型加载器
    //std::string obj_path = "../models/spot/"; // 模型文件所在路径, 相对路径, 起始路径为main 函数所在路径
 	std::string obj_path = "./models/spot/";

    // Load .obj File
	// 将模型从文件加载到内存
	// 打开后可以看到, 这是一个纯白色的牛
    //bool loadout = Loader.LoadFile("../models/spot/spot_triangulated_good.obj"); 
	bool loadout = Loader.LoadFile("./models/spot/spot_triangulated_good.obj");
	// 遍历三角网,  std::vector<Mesh> LoadedMeshes
	// 这里所有的顶点数据都是存储在一个mesh 中的,可能是为了增强代码的健壮性吧, 这里采用了循环遍历
    for(auto mesh:Loader.LoadedMeshes) 
    {
		// static bool printed_p1 = false;
	    // if (!printed_p1)
	    // {
	    //     std::cout << Loader.LoadedMeshes.size() << std::endl; // 1
	    //     printed_p1 = true;
	    // }
		// 遍历顶点数组
		// std::vector<Vertex> Vertices, Vertices 是 Mesh 中的成员, Vertices 中的元素为 struct Vertex
		// Vertex 的成员是 Vector3 Position, Vector3 Normal, Vector2 TextureCoordinate
		// 在这里将依次遍历顶点 Vertex, 一次遍历3个
        for(int i=0;i<mesh.Vertices.size();i+=3)
        {
			// static bool printed_p2 = false;
		    // if (!printed_p2)
		    // {
		    //     std::cout << mesh.Vertices.size() << std::endl;//17568
		    //     printed_p2 = true;
		    // }

            Triangle* t = new Triangle(); // 指针

            for(int j=0;j<3;j++)
            {
                t->setVertex(j,Vector4f(mesh.Vertices[i+j].Position.X,mesh.Vertices[i+j].Position.Y,mesh.Vertices[i+j].Position.Z,1.0));
                t->setNormal(j,Vector3f(mesh.Vertices[i+j].Normal.X,mesh.Vertices[i+j].Normal.Y,mesh.Vertices[i+j].Normal.Z));
                t->setTexCoord(j,Vector2f(mesh.Vertices[i+j].TextureCoordinate.X, mesh.Vertices[i+j].TextureCoordinate.Y));
            }
            TriangleList.push_back(t); // 存入 TriangleList
        }
    }

    rst::rasterizer r(700, 700); // 700 * 700 的 rasterizer 对象
	
	// 纹理文件存放路径, 从后面可以看出, 纹理文件是存放在 obj_path -  "../models/spot/" 中的
	// 默认纹理文件为 hmap.jpg, 打开后可以看到 这是一个 height map
    auto texture_path = "hmap.jpg";

	// Texture(obj_path + texture_path), 一个 Texture 类型的临时对象
	// 在这里指定了 rasterizer 对象中的纹理图像 void set_texture(Texture tex) { texture = tex; }
    r.set_texture(Texture(obj_path + texture_path)); 
	// 一个函数包装器对象, 返回值为 Eigen::Vector3f, 参数为 fragment_shader_payload
	// 这里包装的对象为函数指针, 默认为 phong_fragment_shader
    std::function<Eigen::Vector3f(fragment_shader_payload)> active_shader = phong_fragment_shader;

    if (argc >= 2) // 根据传递给 main 函数的参数不同, 选择不同的 shader
    {
        command_line = true;
        filename = std::string(argv[1]); // 新的输出文件名称

        if (argc == 3 && std::string(argv[2]) == "texture")
        {
            std::cout << "Rasterizing using the texture shader\n";
            active_shader = texture_fragment_shader;
			// 当使用 texture_fragment_shader 时, 更换 纹理图像路径为 "spot_texture.png"
			// 打开后可以看到 这是一个基础纹理
            texture_path = "spot_texture.png"; 
            r.set_texture(Texture(obj_path + texture_path)); // 同时将 rasterizer 对象中的纹理图像也更换了
        }
		// 其它情况就是简单的更换 shader
        else if (argc == 3 && std::string(argv[2]) == "normal")
        {
            std::cout << "Rasterizing using the normal shader\n";
            active_shader = normal_fragment_shader;
        }
        else if (argc == 3 && std::string(argv[2]) == "phong")
        {
            std::cout << "Rasterizing using the phong shader\n";
            active_shader = phong_fragment_shader;
        }
        else if (argc == 3 && std::string(argv[2]) == "bump")
        {
            std::cout << "Rasterizing using the bump shader\n";
            active_shader = bump_fragment_shader;
        }
        else if (argc == 3 && std::string(argv[2]) == "displacement")
        {
            std::cout << "Rasterizing using the bump shader\n";
            active_shader = displacement_fragment_shader;
        }
    }

    Eigen::Vector3f eye_pos = {0,0,10}; // 摄像机/视线 位置坐标

    r.set_vertex_shader(vertex_shader); // 设置 rasterizer 对象的 顶点着色器, 本练习采用的是 phong shading, 所以顶点着色没有做任何处理
    r.set_fragment_shader(active_shader); // 设置 rasterizer 对象的 片元着色器

    int key = 0; // 这个参数本练习用不到
    int frame_count = 0;

    if (command_line)
    {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);// 初始化 rasterizer 对象
        r.set_model(get_model_matrix(angle)); // 设置 rasterizer 对象的 模型变换, 将模型从模型坐标系转换到世界坐标系
        r.set_view(get_view_matrix(eye_pos)); // 设置 rasterizer 对象的 模型变换, 将模型从世界坐标系转换到视图坐标系
        r.set_projection(get_projection_matrix(45.0, 1, 0.1, 50)); // 设置 rasterizer 对象的 投影变换, 将模型从世界坐标系转换到标准设备坐标系

        r.draw(TriangleList);
        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);

        cv::imwrite(filename, image);

        return 0;
    }

// 本练习不需要关注后面的代码
    while(key != 27)
    {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45.0, 1, 0.1, 50));

        //r.draw(pos_id, ind_id, col_id, rst::Primitive::Triangle);
        r.draw(TriangleList);
        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);

        cv::imshow("image", image);
        cv::imwrite(filename, image);
        key = cv::waitKey(10);

        if (key == 'a' )
        {
            angle -= 0.1;
        }
        else if (key == 'd')
        {
            angle += 0.1;
        }

    }
    return 0;
}
