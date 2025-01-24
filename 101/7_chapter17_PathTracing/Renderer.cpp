//
// Created by goksu on 2/25/20.
//

#include <fstream>
#include "Scene.hpp"
#include "Renderer.hpp"
// Devin for threads
#include <thread>
#include <mutex>
#include <vector>
//#include <functional>
//#include <memory>
#include <atomic>


inline float deg2rad(const float& deg) { return deg * M_PI / 180.0; }

const float EPSILON = 0.00001;

// The main render function. This where we iterate over all pixels in the image,
// generate primary rays and cast these rays into the scene. The content of the
// framebuffer is saved to a file.
void Renderer::Render(const Scene& scene)
{
    std::vector<Vector3f> framebuffer(scene.width * scene.height);

    float scale = tan(deg2rad(scene.fov * 0.5));
    float imageAspectRatio = scene.width / (float)scene.height;
    Vector3f eye_pos(278, 273, -800);
    //int m = 0;
    // change the spp value to change sample ammount
   
	int cnt_thread = 20;//std::thread::hardware_concurrency(); // 32
	int spp = 16;

	std::cout << "Count of Thread: " << cnt_thread << "\n";
	std::cout << "SPP: " << spp << "\n";
	//std::mutex mutex_buffer;
	int height_block = scene.height / cnt_thread;
	float unit_progress = 1 / (float)scene.height;
	std::atomic<float> progress = 0.f;
	//float process = 0;
	std::vector<std::thread> threads(cnt_thread);

	auto castRay_ = [&](int start_height, int end_height){
	//auto castRay_ = std::function<void(int&, int&)>([&](int& start_height, int& end_height){
		float local_progress = 0; // 用于当前线程进度计数
	    for (uint32_t j = start_height; j < end_height; ++j) {
	        for (uint32_t i = 0; i < scene.width; ++i) {
	            // generate primary ray direction

	            float x = (2 * (i + 0.5) / (float)scene.width - 1) *
	                      imageAspectRatio * scale;
	            float y = (1 - 2 * (j + 0.5) / (float)scene.height) * scale;

	            Vector3f dir = normalize(Vector3f(-x, y, 1));
				Vector3f color(0.f);
				for(int k = 0; k < spp; ++k){
					color += scene.castRay(Ray(eye_pos, dir), 0);
				}
				framebuffer[j * scene.width + i] = color / spp;
	        }
			local_progress += unit_progress;
	    }
		progress.fetch_add(local_progress);
		UpdateProgress(progress.load());
	};

	for(int t = 0; t < cnt_thread; ++t)
	{
		int start_block = t * height_block;
		int end_block = (t ==  cnt_thread - 1 ? scene.height : start_block + height_block); // 避免无法整除
		//std::make_shared<std::thread>(castRay_, start_block, end_block);
		threads[t] = std::thread(castRay_, start_block, end_block);
	}

	for(auto&thread : threads) {thread.join();}
	
	UpdateProgress(1.f);
    // save framebuffer to file
    FILE* fp = fopen("binary.ppm", "wb");
    (void)fprintf(fp, "P6\n%d %d\n255\n", scene.width, scene.height);
    for (auto i = 0; i < scene.height * scene.width; ++i) {
        static unsigned char color[3];
        color[0] = (unsigned char)(255 * std::pow(clamp(0, 1, framebuffer[i].x), 0.6f));
        color[1] = (unsigned char)(255 * std::pow(clamp(0, 1, framebuffer[i].y), 0.6f));
        color[2] = (unsigned char)(255 * std::pow(clamp(0, 1, framebuffer[i].z), 0.6f));
        fwrite(color, 1, 3, fp);
    }
    fclose(fp);    
}
