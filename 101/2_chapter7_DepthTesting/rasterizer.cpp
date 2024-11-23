// clang-format off
//
// Created by goksu on 4/6/19.
//

#include <algorithm>
#include <vector>
#include "rasterizer.hpp"
#include <opencv2/opencv.hpp>
#include <math.h>
#include <utility>
#include <Windows.h>


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

auto to_vec4(const Eigen::Vector3f& v3, float w = 1.0f)
{
    return Vector4f(v3.x(), v3.y(), v3.z(), w);
}


static bool insideTriangle(int x, int y, const Vector3f* _v)
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

		if (v1.cross(v2).z() < 0)
			return false;
	}
	return true;
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

static std::tuple<float, float, float> computeBarycentric2D(float x, float y, const Vector3f* v)
{
    float c1 = (x*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*y + v[1].x()*v[2].y() - v[2].x()*v[1].y()) / (v[0].x()*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*v[0].y() + v[1].x()*v[2].y() - v[2].x()*v[1].y());
    float c2 = (x*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*y + v[2].x()*v[0].y() - v[0].x()*v[2].y()) / (v[1].x()*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*v[1].y() + v[2].x()*v[0].y() - v[0].x()*v[2].y());
    float c3 = (x*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*y + v[0].x()*v[1].y() - v[1].x()*v[0].y()) / (v[2].x()*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*v[2].y() + v[0].x()*v[1].y() - v[1].x()*v[0].y());
    return {c1,c2,c3};
}

void rst::rasterizer::draw(pos_buf_id pos_buffer, ind_buf_id ind_buffer, col_buf_id col_buffer, Primitive type)
{
    auto& buf = pos_buf[pos_buffer.pos_id]; // 获取(引用) map 键值对的value
    auto& ind = ind_buf[ind_buffer.ind_id]; // 同上
    auto& col = col_buf[col_buffer.col_id]; // 同上

    float f1 = (50 - 0.1) / 2.0;
    float f2 = (50 + 0.1) / 2.0;

    Eigen::Matrix4f mvp = projection * view * model;
    for (auto& i : ind)
    {
        Triangle t;
        Eigen::Vector4f v[] = {
                mvp * to_vec4(buf[i[0]], 1.0f),// buf[{0, 1, 2}[0]] 等效于buf[0], to_vec4(buf[i[0]], 1.0f) 等效于 to_vec4(2, 0, -2, 1.0f)
                mvp * to_vec4(buf[i[1]], 1.0f),// buf[{0, 1, 2}[1]] 等效于buf[1], 同上
                mvp * to_vec4(buf[i[2]], 1.0f) // buf[{0, 1, 2}[2]] 等效于buf[2], 同上
        };
        //Homogeneous division, 齐次化
        for (auto& vec : v) {
            vec /= vec.w();
        }
        //Viewport transformation
        for (auto & vert : v)
        {
			// 因为 v 是经过 mvp 变换的, 其坐标是标准化的, 在[-1, 1]范围, + 1 将其调整到[0, 2]范围
			// 0.5*width*2 则将三角形的范围变换为整个视口范围
            vert.x() = 0.5*width*(vert.x()+1.0);
            vert.y() = 0.5*height*(vert.y()+1.0);
            vert.z() = vert.z() * f1 + f2;// 将z坐标调整(放大 + 平移)至 [0.1, 50] 的范围(避免 z 作为为0)
        }

        for (int i = 0; i < 3; ++i)
        {
            t.setVertex(i, v[i].head<3>()); // .head<3>() 返回一个包含该向量前三个元素的 Eigen::Vector3f 类型的三维向量
            t.setVertex(i, v[i].head<3>());
            t.setVertex(i, v[i].head<3>());
        }


        auto col_x = col[i[0]]; // col[{0, 1, 2}[0]] 等效于col[0], 等效于 {217.0, 238.0, 185.0}
        auto col_y = col[i[1]]; // col[{0, 1, 2}[1]] 等效于col[1], 等效于 {217.0, 238.0, 185.0}
        auto col_z = col[i[2]]; // ... 三个顶点颜色相同

		//设置顶点的颜色
        t.setColor(0, col_x[0], col_x[1], col_x[2]); // 0 表示顶点 编号(索引)
        t.setColor(1, col_y[0], col_y[1], col_y[2]);
        t.setColor(2, col_z[0], col_z[1], col_z[2]);

        rasterize_triangle(t);
    }
}

//Screen space rasterization
void rst::rasterizer::rasterize_triangle(const Triangle& t) {
    auto v = t.toVector4();  // 获取三角形顶点的齐次坐标
    Rectangle rec = GetBoundingBox(t);  // 获取三角形的包围盒

    // 创建临时缓冲区，用于存储 2x2 超采样的颜色和深度
    std::vector<Eigen::Vector3f> frame_buf_temp(4 * frame_buf.size(), Eigen::Vector3f{0, 0, 0});
    std::vector<float> depth_buf_temp(4 * depth_buf.size(), std::numeric_limits<float>::infinity());

    // 遍历包围盒内的像素，使用 2x2 SSAA 进行采样
    for (int y = rec._bot; y < rec._top; ++y) {
        for (int x = rec._left; x < rec._right; ++x) {
            Eigen::Vector3f color_sum(0, 0, 0);  // 颜色累加
            float depth_sum = 0.0f;              // 深度累加
            int samples_in_triangle = 0;         // 计数子像素中在三角形内的数量

            // 2x2 超采样：遍历每个像素中的 4 个子像素
            for (int sub_y = 0; sub_y < 2; ++sub_y) {
                for (int sub_x = 0; sub_x < 2; ++sub_x) {
                    // 计算子像素位置
                    float sample_x = x + (sub_x + 0.5f) / 2.0f;
                    float sample_y = y + (sub_y + 0.5f) / 2.0f;

                    // 检查子像素是否在三角形内
                    if (insideTriangle(sample_x, sample_y, t.v)) {
                        // 在三角形内，增加计数
                        samples_in_triangle++;

                        // 计算子像素的重心坐标并插值深度
                        auto [alpha, beta, gamma] = computeBarycentric2D(sample_x, sample_y, t.v);
                        float w_reciprocal = 1.0f / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                        float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                        z_interpolated *= w_reciprocal;

                        // 获取子像素的全局索引
                        long subpixel_index = ((height - y - 1) * width + x) * 4 + (sub_y * 2 + sub_x);

                        // 深度测试
                        if (z_interpolated < depth_buf_temp[subpixel_index]) {
                            depth_buf_temp[subpixel_index] = z_interpolated;
                            frame_buf_temp[subpixel_index] = t.getColor();
                            color_sum += t.getColor() / 4;
                            depth_sum += z_interpolated / 4;
                        }
                    }
                }
            }

            // 如果有子像素在三角形内，根据子像素数量对颜色和深度求加权平均
            if (samples_in_triangle > 0) {
                color_sum *= (samples_in_triangle / 4.0f);
                depth_sum *= (samples_in_triangle / 4.0f);
                long pixel_index = get_index(x, y);
                frame_buf[pixel_index] = color_sum;
                depth_buf[pixel_index] = depth_sum;
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
}

int rst::rasterizer::get_index(int x, int y)
{
    return (height-1-y)*width + x;
}

void rst::rasterizer::set_pixel(const Eigen::Vector3f& point, const Eigen::Vector3f& color)
{
    //old index: auto ind = point.y() + point.x() * width;
    auto ind = (height-1-point.y())*width + point.x();
    frame_buf[ind] = color;

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

float rst::rasterizer::GetCoverageRatio(int x, int y, const Triangle& t)const
{
	float ratio = 0.f;
	float offsets [4][2] = {{-0.25f, -0.25f}, {+0.25f, -0.25f}, {-0.25f, +0.25f}, {+0.25f, +0.25f}};
	for(const auto& offset : offsets)
	{
		if(insideTriangle((float)x + offset[0], (float)y + offset[1], t.v))
			ratio += 0.25f;
	}
	return ratio;
}

// clang-format on