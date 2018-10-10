#pragma once
#include <opencv2/opencv.hpp>
#include <string>

#include "camera.h"



struct sp_source {
	int label;
	double chi;
};

class SuperPixel     ///单个superpixel区域
{
public:
	cv::Point center;
	std::vector<cv::Point> contour;
	std::vector<cv::Point> pixels;     //包含的像素列表
	std::vector<float> pixels_depth;   //像素的深度列表
	int pixel_num;
	int depth_num;
	float depth_average;
	float depth_max;
	float depth_min;
	~SuperPixel() {}

	int minx, maxx;
	int miny, maxy;

	cv::Mat sp_mat;
	cv::Mat sp_mask;
	cv::MatND hist;
	
	cv::Point& get_pixel(int i);
	void create();
	bool have_depth();
	bool exist_source(int k);

	std::vector<sp_source> source_list;
	std::vector<sp_source> neibour_list;


private:
};



class ImgData
{
public:
	//变量
	int id;
	Camera cam;
	cv::Mat depth_mat;
	cv::Mat world_center;
	cv::Mat origin_img;
	cv::Mat sp_label;
	cv::Mat sp_contour;
	int sp_num;
	std::vector<SuperPixel> data;         //超像素列表
	std::string path_output;
	//函数
	ImgData(int _id, Camera& _cam, cv::Mat& _origin_img, cv::Mat& _depth_mat, cv::Mat& _sp_label, cv::Mat& _sp_contour, int _sp_num);
	ImgData() {}
	~ImgData() {}

	SuperPixel& get_superpixel(int i);
	float& get_pixel_depth(cv::Point& point);

private:
	void create_path();
	void calc_world_center();
	void generate_sp();
	float find_depth(cv::Point& p_begin);
	void save_depth_image();
	void save_sp_image();

	void get_source(int k);
	void sp_dij(int k);
	void get_neibour(int k);

	void save_sp_image_test();
	void find_source_test();
	void update_depth(int k, int a, int b, int c);
};

void mix_pic(std::vector<ImgData>& imgdata_vec, Camera& now_cam, std::vector<int>& img_id, cv::Mat& output_img);
