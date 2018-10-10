#pragma once

#include <opencv2/opencv.hpp>
#include <string>


class Camera
{
public:
	//member
	cv::Mat K, R, T, Ki, Ri;
	cv::Mat world_to_cam, cam_to_world;
	cv::Mat cam_pos;
	//function

	Camera() {}
	Camera(const Camera& cam);
	Camera(cv::Mat& _K, cv::Mat& _R, cv::Mat& _T);
	cv::Mat get_world_pos(cv::Point point, float depth);
	cv::Point get_cam_pos(cv::Mat xyz);
	Camera generate_novel_cam(cv::Mat cam_pos_novel, cv::Mat world_center);
	void fill_reprojection(Camera& des_cam, cv::Mat& mat, cv::Mat& vec);

	void debug();
};

cv::Point cal_reprojection(cv::Point origin_point, float depth, cv::Mat& mat, cv::Mat& vec);
