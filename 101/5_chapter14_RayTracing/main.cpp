#include "Scene.hpp"
#include "Sphere.hpp"
#include "Triangle.hpp"
#include "Light.hpp"
#include "Renderer.hpp"

// In the main function of the program, we create the scene (create objects and lights)
// as well as set the options for the render (image width and height, maximum recursion
// depth, field-of-view, etc.). We then call the render function().
int main()
{
    Scene scene(1280, 960);

    auto sph1 = std::make_unique<Sphere>(Vector3f(-1, 0, -12), 2);// unique_ptr  sph2 指向一个Sphere对象, 半径为 2
    sph1->materialType = DIFFUSE_AND_GLOSSY;// 类似金属抛光那样的材质
    sph1->diffuseColor = Vector3f(0.6, 0.7, 0.8); // 定义漫反射的颜色(这里是比背景色浅一些的蓝色)

    auto sph2 = std::make_unique<Sphere>(Vector3f(0.5, -0.5, -8), 1.5); //同上
    sph2->ior = 1.5; // 折射率
    sph2->materialType = REFLECTION_AND_REFRACTION;//类似玻璃那样的材质, 这里使用默认的颜色(深灰色)

    scene.Add(std::move(sph1));
    scene.Add(std::move(sph2));

    Vector3f verts[4] = {{-5,-3,-6}, {5,-3,-6}, {5,-3,-16}, {-5,-3,-16}};// y = -3, x \in [-5, 5], z \in [-6, -16], 一个矩形平面
    uint32_t vertIndex[6] = {0, 1, 3, 1, 2, 3};// 两个三角形
    Vector2f st[4] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};// 顶点纹理坐标
    auto mesh = std::make_unique<MeshTriangle>(verts, vertIndex, 2, st);
    mesh->materialType = DIFFUSE_AND_GLOSSY;

    scene.Add(std::move(mesh));
    scene.Add(std::make_unique<Light>(Vector3f(-20, 70, 20), 0.5));
    scene.Add(std::make_unique<Light>(Vector3f(30, 50, -12), 0.5));

    Renderer r;
    r.Render(scene);

    return 0;
}