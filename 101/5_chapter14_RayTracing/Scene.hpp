#pragma once

#include <vector>
#include <memory>
#include "Vector.hpp"
#include "Object.hpp"
#include "Light.hpp"

class Scene
{
public:
    // setting up options
    int width = 1280;
    int height = 960;
    double fov = 90;
    Vector3f backgroundColor = Vector3f(0.235294, 0.67451, 0.843137);// 亮蓝色
    int maxDepth = 5; // 递归深度, 最多递归 5 次
    float epsilon = 0.00001;

    Scene(int w, int h) : width(w), height(h)
    {}
	// ~~~~~~~~~~是不是应该这样 std::unique_ptr<Object>&& object 才对? 这里先留着疑问~~~~~~~~
    void Add(std::unique_ptr<Object> object) { objects.push_back(std::move(object)); }
    void Add(std::unique_ptr<Light> light) { lights.push_back(std::move(light)); }
	
	// [[nodiscard]] 表示在调用该函数时,必须有变量存储该返回值, 不能像 返回值为void 的函数那样仅调用
    [[nodiscard]] const std::vector<std::unique_ptr<Object> >& get_objects() const { return objects; }
    [[nodiscard]] const std::vector<std::unique_ptr<Light> >&  get_lights() const { return lights; }

private:
    // creating the scene (adding objects and lights)
    std::vector<std::unique_ptr<Object> > objects;
    std::vector<std::unique_ptr<Light> > lights;
};