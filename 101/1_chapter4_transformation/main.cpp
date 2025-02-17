#include "triangle.hpp"
#include "rasterizer.hpp"
#include <Eigen>
#include <iostream>
#include <opencv2/opencv.hpp>

constexpr double MY_PI = 3.1415926;
float AngleToRadian(float angle)
{
	return (angle / 180) * MY_PI;
}
Eigen::Matrix4f get_view_matrix(Eigen::Vector3f eye_pos)
{
    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();

    Eigen::Matrix4f translate;
    translate << 1, 0, 0, -eye_pos[0], 0, 1, 0, -eye_pos[1], 0, 0, 1,
        -eye_pos[2], 0, 0, 0, 1;

    view = translate * view;

    return view;
}

Eigen::Matrix4f get_model_matrix(float rotation_angle)
{
    Eigen::Matrix4f model = Eigen::Matrix4f::Identity();

    // TODO: Implement this function
    // Create the model matrix for rotating the triangle around the Z axis.
    // Then return it.

	Eigen::Matrix4f rotate;//旋转矩阵 - 1个4 × 4 的零矩阵, 元素类型为float
	float rotate_radian = AngleToRadian(rotation_angle);
	rotate << std::cos(rotate_radian), -std::sin(rotate_radian), 0.f, 0.f,
				std::sin(rotate_radian), std::cos(rotate_radian), 0.f, 0.f,
				0.f, 0.f, 1.f, 0.f,
				0.f, 0.f, 0.f, 1.f;
	model = rotate * model;
	return model;
}

Eigen::Matrix4f get_projection_matrix(float eye_fov, float aspect_ratio,
                                      float zNear, float zFar)
{
    // Students will implement this function

    Eigen::Matrix4f projection = Eigen::Matrix4f::Identity();

    // TODO: Implement this function
    // Create the projection matrix for the given parameters.
    // Then return it.
	//本例给出的 zNear 和 zFar 都是正值(距离), 而推导用的负值(坐标)
	zNear = -zNear;
	zFar *= -1;
	// Squish - Perspactive Protection to Orthographic Protection
	Eigen::Matrix4f squish;
 	squish << zNear, 0, 0, 0,
				0, zNear, 0, 0,
				0, 0, zNear + zFar, -(zNear * zFar),
				0, 0, 1, 0;
	
	// Orthographic Projection
	float yTop, yBot, xLeft,xRight;//需要float, 而不是int
	yBot = zNear * tan(eye_fov / 2);
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
	// perspToOrtho << 
    return projection;
}

int main(int argc, const char** argv)
{
    float angle = 0;
    bool command_line = false;
    std::string filename = "output.png";

    if (argc >= 3) {
        command_line = true;
        angle = std::stof(argv[2]); // -r by default
        if (argc == 4) {
            filename = std::string(argv[3]);
        }
        else
            return 0;
    }

    rst::rasterizer r(700, 700);

    Eigen::Vector3f eye_pos = {0, 0, 5};

    std::vector<Eigen::Vector3f> pos{{2, 0, -2}, {0, 2, -2}, {-2, 0, -2}};

    std::vector<Eigen::Vector3i> ind{{0, 1, 2}};

    auto pos_id = r.load_positions(pos);
    auto ind_id = r.load_indices(ind);

    int key = 0;
    int frame_count = 0;

    if (command_line) {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45, 1, 0.1, 50));

        r.draw(pos_id, ind_id, rst::Primitive::Triangle);
        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);

        cv::imwrite(filename, image);

        return 0;
    }

    while (key != 27) {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45, 1, 0.1, 50));

        r.draw(pos_id, ind_id, rst::Primitive::Triangle);

        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::imshow("image", image);
        key = cv::waitKey(10);

        std::cout << "frame count: " << frame_count++ << '\n';

        if (key == 'a') {
            angle += 10;
        }
        else if (key == 'd') {
            angle -= 10;
        }
    }

    return 0;
}
