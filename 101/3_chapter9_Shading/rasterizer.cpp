//
// Created by goksu on 4/6/19.
//

#include <algorithm>
#include "rasterizer.hpp"
#include <opencv2/opencv.hpp>
#include <math.h>


rst::pos_buf_id rst::rasterizer::load_positions(const std::vector<Eigen::Vector3f> &positions)
{
    auto id = get_next_id();
    pos_buf.emplace(id, positions);

    return {id};
}

rst::ind_buf_id rst::rasterizer::load_indices(const std::vector<Eigen::Vector3i> &indices)
{
    auto id = get_next_id();
    ind_buf.emplace(id, indices);

    return {id};
}

rst::col_buf_id rst::rasterizer::load_colors(const std::vector<Eigen::Vector3f> &cols)
{
    auto id = get_next_id();
    col_buf.emplace(id, cols);

    return {id};
}

rst::col_buf_id rst::rasterizer::load_normals(const std::vector<Eigen::Vector3f>& normals)
{
    auto id = get_next_id();
    nor_buf.emplace(id, normals);

    normal_id = id;

    return {id};
}


// Bresenham's line drawing algorithm
void rst::rasterizer::draw_line(Eigen::Vector3f begin, Eigen::Vector3f end)
{
    auto x1 = begin.x();
    auto y1 = begin.y();
    auto x2 = end.x();
    auto y2 = end.y();

    Eigen::Vector3f line_color = {255, 255, 255};

    int x,y,dx,dy,dx1,dy1,px,py,xe,ye,i;

    dx=x2-x1;
    dy=y2-y1;
    dx1=fabs(dx);
    dy1=fabs(dy);
    px=2*dy1-dx1;
    py=2*dx1-dy1;

    if(dy1<=dx1)
    {
        if(dx>=0)
        {
            x=x1;
            y=y1;
            xe=x2;
        }
        else
        {
            x=x2;
            y=y2;
            xe=x1;
        }
        Eigen::Vector2i point = Eigen::Vector2i(x, y);
        set_pixel(point,line_color);
        for(i=0;x<xe;i++)
        {
            x=x+1;
            if(px<0)
            {
                px=px+2*dy1;
            }
            else
            {
                if((dx<0 && dy<0) || (dx>0 && dy>0))
                {
                    y=y+1;
                }
                else
                {
                    y=y-1;
                }
                px=px+2*(dy1-dx1);
            }
//            delay(0);
            Eigen::Vector2i point = Eigen::Vector2i(x, y);
            set_pixel(point,line_color);
        }
    }
    else
    {
        if(dy>=0)
        {
            x=x1;
            y=y1;
            ye=y2;
        }
        else
        {
            x=x2;
            y=y2;
            ye=y1;
        }
        Eigen::Vector2i point = Eigen::Vector2i(x, y);
        set_pixel(point,line_color);
        for(i=0;y<ye;i++)
        {
            y=y+1;
            if(py<=0)
            {
                py=py+2*dx1;
            }
            else
            {
                if((dx<0 && dy<0) || (dx>0 && dy>0))
                {
                    x=x+1;
                }
                else
                {
                    x=x-1;
                }
                py=py+2*(dx1-dy1);
            }
//            delay(0);
            Eigen::Vector2i point = Eigen::Vector2i(x, y);
            set_pixel(point,line_color);
        }
    }
}

auto to_vec4(const Eigen::Vector3f& v3, float w = 1.0f)
{
    return Vector4f(v3.x(), v3.y(), v3.z(), w);
}

static bool insideTriangle(int x, int y, const Vector4f* _v){
    Vector3f v[3];
    for(int i=0;i<3;i++)
        v[i] = {_v[i].x(),_v[i].y(), 1.0};
    Vector3f f0,f1,f2;
    f0 = v[1].cross(v[0]);
    f1 = v[2].cross(v[1]);
    f2 = v[0].cross(v[2]);
    Vector3f p(x,y,1.);
    if((p.dot(f0)*f0.dot(v[2])>0) && (p.dot(f1)*f1.dot(v[0])>0) && (p.dot(f2)*f2.dot(v[1])>0))
        return true;
    return false;
}

static bool insideTriangle(float x, float y, const Vector3f* _v)//for MSAA, SSAA
{   
    // TODO : Implement this function to check if the point (x, y) is inside the triangle represented by _v[0], _v[1], _v[2]
	Vector3f v1;
	Vector3f v2;
	for(int i = 0; i < 3; ++i)
	{
		int cur = i % 3;
		int next = (i + 1) % 3;

		v1 << (_v[next](0) - _v[cur](0)), (_v[next](1) - _v[cur](1)), 0.f;
		v2 << (x - _v[cur](0)), (y - _v[cur](1)), 0.f;

		if (v1.cross(v2).z() < 0.f)
			return false;
	}
	return true;
}


// static std::tuple<float, float, float> computeBarycentric2D(float x, float y, const Vector4f* v){
//     float c1 = (x*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*y + v[1].x()*v[2].y() - v[2].x()*v[1].y()) / (v[0].x()*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*v[0].y() + v[1].x()*v[2].y() - v[2].x()*v[1].y());
//     float c2 = (x*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*y + v[2].x()*v[0].y() - v[0].x()*v[2].y()) / (v[1].x()*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*v[1].y() + v[2].x()*v[0].y() - v[0].x()*v[2].y());
//     float c3 = (x*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*y + v[0].x()*v[1].y() - v[1].x()*v[0].y()) / (v[2].x()*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*v[2].y() + v[0].x()*v[1].y() - v[1].x()*v[0].y());
//     return {c1,c2,c3};
// }

// 优化
static std::tuple<float, float, float> computeBarycentric2D(float x, float y, const Vector4f* v){
	float denom = (v[0].x()*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*v[0].y() + v[1].x()*v[2].y() - v[2].x()*v[1].y());
    float c1 = (x*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*y + v[1].x()*v[2].y() - v[2].x()*v[1].y()) / denom;
    float c2 = (x*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*y + v[2].x()*v[0].y() - v[0].x()*v[2].y()) / denom;
    float c3 = (x*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*y + v[0].x()*v[1].y() - v[1].x()*v[0].y()) / denom;
    return {c1,c2,c3};
}

void rst::rasterizer::draw(std::vector<Triangle *> &TriangleList) {

    float f1 = (50 - 0.1) / 2.0;
    float f2 = (50 + 0.1) / 2.0;

    Eigen::Matrix4f mvp = projection * view * model;
    for (const auto& t:TriangleList) // TriangleList 存储的是从模型中提取的所有 mesh 的三角形
    {
        Triangle newtri = *t; // 将当前三角形保存在 newtri 中

        std::array<Eigen::Vector4f, 3> mm { // 将经过模型变换, 视图变换, 在  ### 视图空间 ###的顶点数据保存在 mm 中
                (view * model * t->v[0]),
                (view * model * t->v[1]),
                (view * model * t->v[2])
        };

        std::array<Eigen::Vector3f, 3> viewspace_pos; 
		// 将 mm 中的数据, 仅取前3个分量(去掉齐次项) 保存在 viewspace_pos 中
        std::transform(mm.begin(), mm.end(), viewspace_pos.begin(), [](auto& v) {return v.template head<3>();});

        Eigen::Vector4f v[] = { // 将经过 模型, 视图, 投影变换, 在 ### 标准化设备坐标 ### 中的顶点数据, 保存在 v 中
                mvp * t->v[0],
                mvp * t->v[1],
                mvp * t->v[2]
        };

        //Homogeneous division, 经过投影变换后, 齐次坐标将被缩放, w不再为1, 因此这里要做齐次坐标标准化
        for (auto& vec : v) {
            vec.x()/=vec.w();
            vec.y()/=vec.w();
            vec.z()/=vec.w();
        }

		/* 模型已经被转换到视图空间，所以法线也需要进行相同的转换
		 * 直接应用 view * model 矩阵并不适用于法线的转换，这是因为法线的变换与顶点的变换有所不同, 法线需要使用逆转置矩阵进行变换：
		 * ** 保持法线的正确方向：在进行非均匀缩放（例如在模型变换中）时，法线的方向可能会发生变化。使用逆转置矩阵可以纠正这种变化
		 * ** 法线的变换规则：法线是向量，而不是点。对于向量的变换，使用逆转置矩阵可以确保法线在变换过程中保持正交性和单位长度
		 */
        Eigen::Matrix4f inv_trans = (view * model).inverse().transpose(); 
        Eigen::Vector4f n[] = {
                inv_trans * to_vec4(t->normal[0], 0.0f),
                inv_trans * to_vec4(t->normal[1], 0.0f),
                inv_trans * to_vec4(t->normal[2], 0.0f)
        };

        //Viewport transformation, 视口变换
        for (auto & vert : v) // 将 v 转换到 屏幕空间中
        {
			// ### 标准设备坐标系 ### 中的xy 在[-1, 1]范围, + 1 将其调整到[0, 2]范围
			// 0.5 * width * 2, 将模型从标准化设备坐标系 变换至 屏幕坐标系中
            vert.x() = 0.5*width*(vert.x()+1.0);
            vert.y() = 0.5*height*(vert.y()+1.0);
			// 在投影变换, 转换到 ### 标准设备坐标系 ### 时, z 值被压缩到了[-1, 1], 这里还原 z 值为初始值
			// 将z坐标调整(放大 + 平移)至 [0.1, 50] 的范围
            vert.z() = vert.z() * f1 + f2;
        }

		// v 现在已经在 ### 屏幕坐标系 ### 了
		// 将 v 的顶点数据, 放到 ~~~ newtri(Triangle) ~~~ 的成员变量  Vector4f v[3] 中
        for (int i = 0; i < 3; ++i)
        {
            //screen space coordinates
            newtri.setVertex(i, v[i]);
        }
		// 将最中处理完成的 n, 即法线, 放到 ~~~ newtri(Triangle) ~~~ 的成员变量  Vector3f normal[3] 中
        for (int i = 0; i < 3; ++i)
        {
            //view space normal
            newtri.setNormal(i, n[i].head<3>()); 
        }

		// 将 顶点颜色 放到 ~~~ newtri(Triangle) ~~~ 的成员变量 Vector3f color[3] 中
		// 这里给定的顶点颜色是 一个棕色
        newtri.setColor(0, 148,121.0,92.0); 
        newtri.setColor(1, 148,121.0,92.0);
        newtri.setColor(2, 148,121.0,92.0);

        // Also pass view space vertice position
		/* newtri 的顶点坐标(含齐次项)
		 * ** 顶点的 xyz 坐标经过了模型, 视图, 投影, 视口 变换, 所以是在屏幕坐标系中的
		 * ** z 坐标做了特殊处理, 使其在数值上恢复为了 投影变换前的z值
		 * newtri 的法线经过了 模型, 视图变换
		 * viewspace_pos 是经过了 模型,视图变换 的顶点坐标(不含齐次项)+
		 * 可以理解为这两个参数是模型上的同一个点, 前者是屏幕空间的点, 后者是视图空间中的点
		 */
        rasterize_triangle(newtri, viewspace_pos);
    }
}

static Eigen::Vector3f interpolate(float alpha, float beta, float gamma, const Eigen::Vector3f& vert1, const Eigen::Vector3f& vert2, const Eigen::Vector3f& vert3, float weight)
{
    return (alpha * vert1 + beta * vert2 + gamma * vert3) / weight;
}

static Eigen::Vector2f interpolate(float alpha, float beta, float gamma, const Eigen::Vector2f& vert1, const Eigen::Vector2f& vert2, const Eigen::Vector2f& vert3, float weight)
{
    auto u = (alpha * vert1[0] + beta * vert2[0] + gamma * vert3[0]);
    auto v = (alpha * vert1[1] + beta * vert2[1] + gamma * vert3[1]);

    u /= weight;
    v /= weight;

    return Eigen::Vector2f(u, v);
}

//Screen space rasterization
void rst::rasterizer::rasterize_triangle(const Triangle& t, const std::array<Eigen::Vector3f, 3>& view_pos) 
{
    // TODO: From your HW3, get the triangle rasterization code.
    // TODO: Inside your rasterization loop:
    //    * v[i].w() is the vertex view space depth value z.
    //    * Z is interpolated view space depth for the current pixel
    //    * zp is depth between zNear and zFar, used for z-buffer

    // float Z = 1.0 / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
    // float zp = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
    // zp *= Z;

    // TODO: Interpolate the attributes:
    // auto interpolated_color
    // auto interpolated_normal
    // auto interpolated_texcoords
    // auto interpolated_shadingcoords

    // Use: fragment_shader_payload payload( interpolated_color, interpolated_normal.normalized(), interpolated_texcoords, texture ? &*texture : nullptr);
    // Use: payload.view_pos = interpolated_shadingcoords;
    // Use: Instead of passing the triangle's color directly to the frame buffer, pass the color to the shaders first to get the final color;
    // Use: auto pixel_color = fragment_shader(payload);
	
	auto v = t.toVector4();  // 获取三角形顶点的齐次坐标
	const Eigen::Vector3f *col = t.color;  // 获取三角形顶点的颜色
	const Eigen::Vector2f *tex_coor = t.tex_coords;  // 获取三角形顶点的纹理坐标
	const Eigen::Vector3f *nor = t.normal;  // 获取三角形顶点的法线

	Rectangle rec = GetBoundingBox(t);
	for(int y = rec._bot; y < rec._top; ++y)
	{
		for(int x = rec._left; x < rec._right; ++x)
		{
			if(insideTriangle(x, y, t.v))// 判断子采样点是否在三角形内
			{
                auto [alpha, beta, gamma] = computeBarycentric2D(x, y, t.v);
                float w_reciprocal = 1.0f / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
				
				Eigen::Vector3f col_interpolated =
					alpha * col[0] / v[0].w() + beta * col[1] / v[1].w() + gamma * col[2] / v[2].w(); // 颜色插值
				Eigen::Vector2f tex_coor_interpolated =
					alpha * tex_coor[0] / v[0].w() + beta * tex_coor[1] / v[1].w() + gamma * tex_coor[2] / v[2].w(); // 纹理插值
				// if(tex_coor_interpolated.x() < 0 || tex_coor_interpolated.x() > 1 || tex_coor_interpolated.y() < 0 || tex_coor_interpolated.y() > 1)
				// 	std::cout << tex_coor_interpolated.x() << ", " << tex_coor_interpolated.y() << std::endl;
				if(tex_coor_interpolated.x() < 0) tex_coor_interpolated.x() = 0;
				if(tex_coor_interpolated.x() > 1) tex_coor_interpolated.x() = 1;
				if(tex_coor_interpolated.y() < 0) tex_coor_interpolated.y() = 0;
				if(tex_coor_interpolated.y() > 1) tex_coor_interpolated.y() = 1;
				Eigen::Vector3f nor_interpolated =
					alpha * nor[0] / v[0].w() + beta * nor[1] / v[1].w() + gamma * nor[2] / v[2].w(); // 法线插值
				Eigen::Vector3f view_pos_interpolated = 
					alpha *view_pos[0] / v[0].w() + beta * view_pos[1] / v[1].w() + gamma * view_pos[2] / v[2].w(); // 视空间位置插值

				z_interpolated *= w_reciprocal;
				col_interpolated *= w_reciprocal;
				tex_coor_interpolated *= w_reciprocal;
				nor_interpolated *= w_reciprocal;
				view_pos_interpolated *= w_reciprocal;
				
				// 计算 子像素在临时缓冲区中的索引
				//float index = ((height - y - 1 ) * width + x);
				float index = (height - y) * width + x;
				if( z_interpolated < depth_buf[index])
				{
					depth_buf[index] = z_interpolated;
					// 先把z 法线 纹理坐标的数据都放到 payload 中
					// texture 是 rst 中的一个成员变量, 存储的是纹理图片
					// 在main 函数中通过  r.set_texture(Texture(obj_path + texture_path));  指定了纹理图片
					fragment_shader_payload payload(col_interpolated, nor_interpolated.normalized(), 
								tex_coor_interpolated, texture ? &*texture : nullptr);
					payload.view_pos = view_pos_interpolated; // 将点在视空间中的位置也放在 payload 中
					auto color = fragment_shader(payload); // fragment_shader 是类 rst 中的一个函数包装器成员变量
					frame_buf[index] = color;
				}
				
			}	
		}
	}
}


void rst::rasterizer::set_model(const Eigen::Matrix4f& m)
{
    model = m;
}

void rst::rasterizer::set_view(const Eigen::Matrix4f& v)
{
    view = v;
}

void rst::rasterizer::set_projection(const Eigen::Matrix4f& p)
{
    projection = p;
}

void rst::rasterizer::clear(rst::Buffers buff)
{
    if ((buff & rst::Buffers::Color) == rst::Buffers::Color)
    {
        std::fill(frame_buf.begin(), frame_buf.end(), Eigen::Vector3f{0, 0, 0});
    }
    if ((buff & rst::Buffers::Depth) == rst::Buffers::Depth)
    {
        std::fill(depth_buf.begin(), depth_buf.end(), std::numeric_limits<float>::infinity());
    }
}

rst::rasterizer::rasterizer(int w, int h) : width(w), height(h)
{
    frame_buf.resize(w * h);
    depth_buf.resize(w * h);

    texture = std::nullopt;
} 

int rst::rasterizer::get_index(int x, int y)
{
    return (height-y)*width + x;
}

void rst::rasterizer::set_pixel(const Vector2i &point, const Eigen::Vector3f &color)
{
    //old index: auto ind = point.y() + point.x() * width;
    int ind = (height-point.y())*width + point.x();
    frame_buf[ind] = color;
}

void rst::rasterizer::set_vertex_shader(std::function<Eigen::Vector3f(vertex_shader_payload)> vert_shader)
{
    vertex_shader = vert_shader;
}

void rst::rasterizer::set_fragment_shader(std::function<Eigen::Vector3f(fragment_shader_payload)> frag_shader)
{
    fragment_shader = frag_shader;
}

typename rst::Rectangle rst::rasterizer::GetBoundingBox(const Triangle& t)const
{
	//std::cout << " " << t.v[0](0) << " " << t.v[1](0) << " " << t.v[2](0)<< std::endl;
	Rectangle rec;

	rec._top = std::max({t.v[0](1),t.v[1](1),t.v[2](1)});
	rec._bot = std::min({t.v[0](1),t.v[1](1),t.v[2](1)});
	rec._right = std::max({t.v[0](0),t.v[1](0),t.v[2](0)});
	rec._left = std::min({t.v[0](0),t.v[1](0),t.v[2](0)});
	return rec;
}