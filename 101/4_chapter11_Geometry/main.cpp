#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>

std::vector<cv::Point2f> control_points;

void mouse_handler(int event, int x, int y, int flags, void *userdata) 
{
    if (event == cv::EVENT_LBUTTONDOWN && control_points.size() < 4) 
    {
        std::cout << "Left button of the mouse is clicked - position (" << x << ", "
        << y << ")" << '\n';
        control_points.emplace_back(x, y);// emplace_back 自动构造对象并 push_back 到 control_points
    }
}

void naive_bezier(const std::vector<cv::Point2f> &points, cv::Mat &window) 
{
    auto &p_0 = points[0];
    auto &p_1 = points[1];
    auto &p_2 = points[2];
    auto &p_3 = points[3];

    for (double t = 0.0; t <= 1.0; t += 0.001) 
    {
        auto point = std::pow(1 - t, 3) * p_0 + 3 * t * std::pow(1 - t, 2) * p_1 +
                 3 * std::pow(t, 2) * (1 - t) * p_2 + std::pow(t, 3) * p_3;

        window.at<cv::Vec3b>(point.y, point.x)[2] = 255;
    }
}

// 使用定义(伯恩斯坦多项式)绘制 Bezier 曲线
float BinomCoeffe(size_t n, size_t k)
{
	return std::tgamma(n + 1) / (std::tgamma(k + 1) *  std::tgamma(n - k + 1)); // ! / ((n-k)! * (k!))
}

cv::Point2f recursive_bezier(const std::vector<cv::Point2f> &control_points, float t)
{
	size_t num = control_points.size() - 1; // 末端点的编号, 起始编号为0
	cv::Point2f ptAtt(0, 0); // Bezier 曲线在参数为 t 时的点
	for(int i = 0; i <= num; ++i)
	{
		float bezierBasis= BinomCoeffe(num, i) * std::pow(t, i) * std::pow(1 - t, num - i); // C^i_n * t^i * (1-t)^(n-i)
		ptAtt += control_points[i] * bezierBasis;
	}
    return ptAtt;
}

void bezier(const std::vector<cv::Point2f> &control_points, cv::Mat &window) 
{
    // TODO: Iterate through all t = 0 to t = 1 with small steps, and call de Casteljau's 
    // recursive Bezier algorithm.
	for (double t = 0.0; t <= 1.0; t += 0.001) 
    {
        cv::Point2f point =  recursive_bezier(control_points, t);
        window.at<cv::Vec3b>(point.y, point.x)[1] = 255;
    }
}

int main() 
{
    cv::Mat window = cv::Mat(700, 700, CV_8UC3, cv::Scalar(0));
    cv::cvtColor(window, window, cv::COLOR_BGR2RGB);
    cv::namedWindow("Bezier Curve", cv::WINDOW_AUTOSIZE);

    cv::setMouseCallback("Bezier Curve", mouse_handler, nullptr);

    int key = -1;
    while (key != 27) 
    {
        for (auto &point : control_points) 
        {
            cv::circle(window, point, 3, {255, 255, 255}, 3);
        }

        if (control_points.size() == 4) 
        {
            //naive_bezier(control_points, window);
            bezier(control_points, window);

            cv::imshow("Bezier Curve", window);
            cv::imwrite("my_bezier_curve.png", window);
            key = cv::waitKey(0);

            return 0;
        }

        cv::imshow("Bezier Curve", window);
        key = cv::waitKey(20);
    }

return 0;
}
