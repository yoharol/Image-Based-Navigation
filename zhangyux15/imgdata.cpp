#include "imgdata.h"

#include "global_var.h"
#include "my_math_tool.h"

#include <io.h>  
#include <direct.h>
#include <thread>

#include <string>
#include <opencv2/imgproc.hpp>
#include <Eigen/Eigen>


using namespace std;
using namespace cv;

#define ave(a,b,c) (a+b+c)/3


Point& SuperPixel::get_pixel(int i)
{
	return pixels[i];
}

bool SuperPixel::have_depth()
{
	return 1.0 * depth_num / pixel_num > 0.05;
}

ImgData::ImgData(int _id, Camera& _cam, Mat& _origin_img, Mat& _depth_mat, Mat& _sp_label, Mat& _sp_contour, int _sp_num) 
{
	cout << "--Init ImgData with file" << to_string(_id) << "..." << endl;			// 不需要重新计算超像素分割
	id = _id;
	cam = _cam;
	sp_num = _sp_num;
	_sp_label.copyTo(sp_label);
	_sp_contour.copyTo(sp_contour);
	_origin_img.copyTo(origin_img);
	_depth_mat.copyTo(depth_mat);
	path_output = PATH_MY_OUTPUT + "\\" + to_string(id);
	create_path();               //创建路径
	calc_world_center();         //计算世界坐标中心
	generate_sp();               //创建超像素列表

	{
		//save_sp_image_test();//功能试探：创建三个超像素局部图
		//find_source_test();//功能试探，找出一个像素的所有类似超像素
		//创建每个超像素的局部类、mask和lab直方图
		//创建超像素列表的相邻像素表
		//创建相邻像素表的相似度
		//get_source(993);
		//system("pause");
	}

	save_depth_image();          //保存深度图
	save_sp_image();             //保存超像素图
}

void ImgData::save_sp_image_test() {
	Mat sp_mat0 = Mat(HEIGHT, WIDTH, CV_8UC3, Scalar(255, 255, 255));
	Mat sp_mask0 = Mat(HEIGHT, WIDTH, CV_8U, Scalar(0));

	int a=993, b=993;

	for (int k = a; k <= b; k++) {
		for (int i = 0; i < get_superpixel(k).pixels.size(); i++) {
			cv::Point p = get_superpixel(k).pixels.at(i);
			Vec3b BGR = origin_img.at<Vec3b>(p);
			cv::circle(sp_mat0, p, 0, CV_RGB((int)BGR[2], (int)BGR[1], (int)BGR[0]));
			cv::circle(sp_mask0, p, 0, Scalar(255));
		}
	}
	imwrite(path_output + "\\superpixel_mask0.png", sp_mat0);
	system("pause");


	/*Mat sp_mat0 = Mat(HEIGHT, WIDTH, CV_8UC3, Scalar(255, 255, 255));
	Mat sp_mat1 = Mat(HEIGHT, WIDTH, CV_8UC3, Scalar(255, 255, 255));
	Mat sp_mat2 = Mat(HEIGHT, WIDTH, CV_8UC3, Scalar(255, 255, 255));

	Mat sp_mask0 = Mat(HEIGHT, WIDTH, CV_8U, Scalar(0));
	Mat sp_mask1 = Mat(HEIGHT, WIDTH, CV_8U, Scalar(0));
	Mat sp_mask2 = Mat(HEIGHT, WIDTH, CV_8U, Scalar(0));

	int m[3] = { 1000,1600,1900 };

	int k = m[0];
	const int channels[] = {0,1,2};
	float range[] = {0,255};
	const float *ranges[] = { range,range,range };
	//const int histsize = 8;                         //直方图精度
	const int histsize[] = { 8,8,8 };

	for (int i = 0; i < get_superpixel(k).pixels.size(); i++) {
		cv::Point p = get_superpixel(k).pixels.at(i);
		Vec3b BGR = origin_img.at<Vec3b>(p);
		cv::circle(sp_mat0, p, 0, CV_RGB((int)BGR[2], (int)BGR[1], (int)BGR[0]));
		cv::circle(sp_mask0, p, 0, Scalar(255));
	}
	//cv::cvtColor(sp_mat0,sp_mat0,cv::COLOR_BGR2Lab);
	MatND hist0;
	calcHist(&sp_mat0, 1, channels, sp_mask0, hist0, 3, histsize, ranges, true, false);
	normalize(hist0, hist0, 0, 1, NORM_MINMAX, -1, Mat());


	k = m[1];
	for (int i = 0; i < get_superpixel(k).pixels.size(); i++) {
		cv::Point p = get_superpixel(k).pixels.at(i);
		Vec3b BGR = origin_img.at<Vec3b>(p);
		cv::circle(sp_mat1, p, 0, CV_RGB((int)BGR[2], (int)BGR[1], (int)BGR[0]));
		cv::circle(sp_mask1, p, 0, Scalar(255));
	}
	//cv::cvtColor(sp_mat1, sp_mat1, cv::COLOR_BGR2Lab);
	MatND hist1;
	calcHist(&sp_mat1, 1, channels, sp_mask1, hist1, 3, histsize, ranges, true, false); 
	normalize(hist1, hist1, 0, 1, NORM_MINMAX, -1, Mat());
	

	k = m[2];
	for (int i = 0; i < get_superpixel(k).pixels.size(); i++) {
		cv::Point p = get_superpixel(k).pixels.at(i);
		Vec3b BGR = origin_img.at<Vec3b>(p);
		cv::circle(sp_mat2, p, 0, CV_RGB((int)BGR[2], (int)BGR[1], (int)BGR[0]));
		cv::circle(sp_mask2, p, 0, Scalar(255));
	}
	//cv::cvtColor(sp_mat2, sp_mat2, cv::COLOR_BGR2Lab);
	MatND hist2;
	calcHist(&sp_mat2, 1, channels, sp_mask2, hist2, 3, histsize, ranges, true, false);
	normalize(hist2, hist2, 0, 1, NORM_MINMAX, -1, Mat());

	//imwrite(path_output + "\\superpixel_origin.png", origin_img);
	imwrite(path_output + "\\superpixel_mask0.png", sp_mat0);
	imwrite(path_output + "\\superpixel_mask1.png", sp_mat1);
	imwrite(path_output + "\\superpixel_mask2.png", sp_mat2);

	double kf1 = compareHist(hist0, hist0, CV_COMP_CHISQR);
	double kf2 = compareHist(hist0, hist1, CV_COMP_CHISQR);
	double kf3 = compareHist(hist0, hist2, CV_COMP_CHISQR);
	double kf4 = compareHist(hist1, hist2, CV_COMP_CHISQR);
	cout << kf1 << ' ' << kf2 << ' ' << kf3 << ' ' << kf4 << '\n';*/
	system("pause");
}

void ImgData::find_source_test() {
	int k = 993;
	Mat R;
	cv::cvtColor(get_superpixel(k).sp_mat, R, cv::COLOR_Lab2BGR);
	imwrite(path_output + "\\superpixel_mask" + to_string(k) + ".png", R);
	Mat sp_mat0 = Mat(HEIGHT, WIDTH, CV_8UC3, Scalar(255, 255, 255));

	for (int i = 0; i < data.size(); i++) {
		if (i == k)
			continue;
		double kf = compareHist(get_superpixel(k).hist, get_superpixel(i).hist, CV_COMP_CHISQR);
		if (kf < 0.5) {
			cv::cvtColor(get_superpixel(i).sp_mat, R, cv::COLOR_Lab2BGR);
			for (int j = 0; j < get_superpixel(i).pixels.size(); j++) {
				cv::Point p = get_superpixel(i).pixels.at(j);
				Vec3b BGR = origin_img.at<Vec3b>(p);
				cv::circle(sp_mat0, p, 0, CV_RGB((int)BGR[2], (int)BGR[1], (int)BGR[0]));
			}
			
		}
	}
	imwrite(path_output + "\\superpixel_mask.png", sp_mat0);
	system("pause");
}

void SuperPixel::create()
{
	pixel_num = pixels.size();
	// 计算中心点和平均深度
	Point point_sum(0, 0);
	float depth_sum = 0;
	depth_min = FLT_MAX;
	depth_max = 0;
	depth_num = 0;
	for (int i = 0; i < pixel_num; i++)
	{
		point_sum += pixels[i];
		float depth = pixels_depth[i];
		if (!is_zero(depth))		 //对于大于1e-6的点记为有depth，猜测无深度时会记为0
		{
			depth_num++;
			depth_sum += depth;
			depth_min = depth < depth_min ? depth : depth_min;
			depth_max = depth > depth_max ? depth : depth_max;
		}
	}

	center = point_sum / pixel_num;
	if ((float)depth_num/(float)pixel_num<0.05)
		depth_average = 0;
	else
		depth_average = depth_sum / depth_num;

	

	//生成mask
	Mat mask = Mat::zeros(HEIGHT, WIDTH, CV_8UC1);
	for (int i = 0; i < pixels.size(); i++)
	{
		Point& temp_pixel = get_pixel(i);
		mask.at<uchar>(temp_pixel) = 1;
	}
	// 计算轮廓并记录到contour
	vector<vector<Point>> temp_contour;
	findContours(mask, temp_contour, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	contour = temp_contour[0];
}

void ImgData::save_depth_image()
{
	cout << "--save_depth_image..." << std::flush;
	// 用彩色表示深度图
	Mat hue_mat;		// 映射到色相0~360
	normalize(depth_mat, hue_mat, 255.0, 0, NORM_MINMAX);

	Mat hsv_pic(HEIGHT, WIDTH, CV_8UC3);
	for (int x = 0; x < WIDTH; x++)
	{
		for (int y = 0; y < HEIGHT; y++)
		{
			Vec3b color{unsigned char(int(hue_mat.at<float>(Point(x, y)))), 100, 255 };
			hsv_pic.at<Vec3b>(Point(x, y)) = color;
		}
	}
	cvtColor(hsv_pic, hsv_pic, CV_HSV2BGR);			// 转换为BGR空间
	imwrite(path_output + "\\depth_map.png", hsv_pic);

	//cout << "save_depth_txt..." << std::flush;									   //输出深度具体数值列表
	/*String filename = path_output + "\\depth.txt";
	fstream file;
	file.open(filename, ofstream::out);
	for (int x = 0; x < WIDTH; x++) {
		for (int y = 0; y < HEIGHT; y++) {
			float depth_n = depth_mat.at<float>(Point(x, y));
			file << depth_n << " ";
		}
		file << "\n";
	}
	file << depth_mat;
	//cout << depth_mat.cols << "-" << WIDTH << " " << depth_mat.rows << "-" << HEIGHT << "\n";
	file.close();*/
	//cout << "OK" << endl;
}

void ImgData::save_sp_image()      //存储超像素分割图
{
	cout << "--save_superpixel_image..." << std::flush;
	Mat sp_img = origin_img.clone();
	sp_img.setTo(Scalar(255, 255, 255), sp_contour);
	imwrite(path_output + "\\superpixel.png", sp_img);

	/*string filename = path_output + "\\superpixel.txt";
	fstream file;
	file.open(filename,ofstream::out);
	//FileStorage fs(filename, FileStorage::WRITE);
	std::cout << "save_superpixel_txt..." << std::flush;
	for (int x = 0; x < WIDTH; x++) {
		for (int y = 0; y < HEIGHT; y++) {
			file << sp_label.at<int>(Point(x, y)) << "," << sp_contour.at<int>(Point(x, y)) << "\t";
		}
		file << "\n";
	}
	file << sp_label;
	file << "\n";
	file << sp_contour;
	file.close();*/
	cout << "OK" << endl;
}

void ImgData::create_path()       
{
	cout << "--create_path " << path_output << "..." << std::flush;
	if (_access(path_output.c_str(), 0) == 0)           //若存在则清空路径
	{
		string command = "rd /s/q " + path_output;
		system(command.c_str());
	}
	_mkdir(path_output.c_str());
	cout << "OK" << endl;
}

void ImgData::generate_sp()
{
	// 把每个点加入到超像素对象中，对深度不足的超像素进行深度更新
	data.resize(sp_num);
	for (int i = 0; i < sp_num; i++) {
		get_superpixel(i).minx = WIDTH;
		get_superpixel(i).maxx = 0;
		get_superpixel(i).miny = HEIGHT;
		get_superpixel(i).maxy = 0;
	}
	for (int x = 0; x < WIDTH; x++)
	{
		for (int y = 0; y < HEIGHT; y++)
		{
			int superpixel_rank = sp_label.at<int>(Point(x, y));
			get_superpixel(superpixel_rank).pixels.push_back(Point(x, y));
			get_superpixel(superpixel_rank).pixels_depth.push_back(depth_mat.at<float>(Point(x, y)));
			if (x > get_superpixel(superpixel_rank).maxx)
				get_superpixel(superpixel_rank).maxx = x;
			if (y > get_superpixel(superpixel_rank).maxy)
				get_superpixel(superpixel_rank).maxy = y;
			if (x < get_superpixel(superpixel_rank).minx)
				get_superpixel(superpixel_rank).minx = x;
			if (y < get_superpixel(superpixel_rank).miny)
				get_superpixel(superpixel_rank).miny = y;
		}
	}

	const int channels[] = { 0,1,2 };
	const float range[] = { 0,255 };
	const float *ranges[] = { range,range,range };
	const int histsize[] = { 8,8,8 };

	for (int k = 0; k < data.size(); k++)
	{
		{
			//创建每一个超像素的局部图、蒙版与直方图
			get_superpixel(k).sp_mat = Mat(get_superpixel(k).maxy- get_superpixel(k).miny, get_superpixel(k).maxx - get_superpixel(k).minx, CV_8UC3, Scalar(255, 255, 255));
			get_superpixel(k).sp_mask = Mat(get_superpixel(k).maxy - get_superpixel(k).miny, get_superpixel(k).maxx - get_superpixel(k).minx, CV_8U, Scalar(0));

			for (int i = 0; i < get_superpixel(k).pixels.size(); i++) {
				cv::Point p = get_superpixel(k).pixels.at(i);
				Vec3b BGR = origin_img.at<Vec3b>(p);
				p.x = p.x - get_superpixel(k).minx;
				p.y = p.y - get_superpixel(k).miny;
				cv::circle(get_superpixel(k).sp_mat, p, 0, CV_RGB((int)BGR[2], (int)BGR[1], (int)BGR[0]));
				cv::circle(get_superpixel(k).sp_mask, p, 0, Scalar(255));
			}
			cv::cvtColor(get_superpixel(k).sp_mat, get_superpixel(k).sp_mat, cv::COLOR_BGR2Lab);
			MatND hist0;
			calcHist(&get_superpixel(k).sp_mat, 1, channels, get_superpixel(k).sp_mask, hist0, 3, histsize, ranges, true, false);
			normalize(hist0, hist0, 0, 1, NORM_MINMAX, -1, Mat());
			get_superpixel(k).hist = hist0;
			/*if (k == 993) {
				imwrite(path_output + "\\superpixel_mat0.png", data[k].sp_mat);
				imwrite(path_output + "\\superpixel_mask0.png", data[k].sp_mask);
				system("pause");
			}*/

			//if (k % 500 == 0) {
			//	imwrite(path_output + "\\superpixel_mask"+to_string(k)+".png", get_superpixel(k).sp_mat);
			//}
		}
		data[k].create();
	}

	for (int k = 0; k < data.size(); k++) {
		if (get_superpixel(k).depth_average == 0) {
			get_source(k);
		}
	 }
}

void ImgData::get_source(int k) {
	//得到超像素k的相似超像素列表，后接dijstra算法
	//cout << "find source ing for " << k << "...\n";
	int max_list = 40;
	bool full = false;
	for (int i = 0; i < data.size(); i++) {
		if (i == k)
			continue;
		if (get_superpixel(i).depth_average < 1e-6)//排除无深度的源像素
			continue;
		double chis = compareHist(get_superpixel(k).hist, get_superpixel(i).hist, CV_COMP_CHISQR);
		if (chis > 10)
			continue;
		if (full) {
			for (int j = 0; j < get_superpixel(k).source_list.size(); j++) {
				if (get_superpixel(k).source_list.at(j).chi > chis) {
					get_superpixel(k).source_list.erase(get_superpixel(k).source_list.begin()+j);
					sp_source temp;
					temp.label = i;
					temp.chi = chis;
					get_superpixel(k).source_list.push_back(temp);
					break;
				}
			}
		}
		else {
			sp_source temp;
			temp.label = i;
			temp.chi = chis;
			get_superpixel(k).source_list.push_back(temp);
			if (get_superpixel(k).source_list.size() == max_list)
				full = true;
		}
	}
	if (get_superpixel(k).source_list.size() > 3)
		sp_dij(k);
}

void ImgData::sp_dij(int k) {
	//cout << "find dijrce ing for " << k << "...\n";
	//找到超像素k最近的3个source像素，并更新
	queue<int> queue_p;
	vector<int> map;
	double *road = new double[sp_num];
	for (int i = 0; i < sp_num; i++) {
		road[i] = 100000.0;
	}
	road[k] = 0;
	queue_p.push(k);
	map.push_back(k);
	get_neibour(k);
	//point_r[k] = true;
	int a, b, c;
	a = b = c = 100;
	double minroad = 100000.0;
	while (!queue_p.empty()) {
		int t = queue_p.front();
		queue_p.pop();
		//cout << t << "\n";
		if (t != k) {
			for (int i = 0; i < get_superpixel(t).neibour_list.size(); i++) {
				if (road[get_superpixel(t).neibour_list.at(i).label] + get_superpixel(t).neibour_list.at(i).chi < road[t]) {
					//cout << "from " << k << " to " << get_superpixel(t).neibour_list.at(i).label << " to " << t <<" is "<< road[get_superpixel(t).neibour_list.at(i).label] <<" + "<< get_superpixel(t).neibour_list.at(i).chi << "\n";
					road[t] = road[get_superpixel(t).neibour_list.at(i).label] + get_superpixel(t).neibour_list.at(i).chi;
				}
			}
			//cout << "road to t is haha " << road[t] << "\n";
			
			if (get_superpixel(k).exist_source(t)) {
				//cout << "get source " << t << " and abc:" << a << b << c;
				if (road[t] < road[a]) {
					c = b;
					b = a;
					a = t;
				}
				else if (road[t] < road[b]) {
					c = b;
					b = t;
				}
				else if (road[t] < road[c]) {
					c = t;
					minroad = road[t];
				}
			}
		}
		if (road[t] <= minroad) {
			for (int i = 0; i < get_superpixel(t).neibour_list.size(); i++) {
				bool exi = false;
				for (int j = 0; j < map.size(); j++) {
					if (map.at(j) == get_superpixel(t).neibour_list.at(i).label) {
						exi = true;
						break;
					}
				}
				if (!exi) {
					queue_p.push(get_superpixel(t).neibour_list.at(i).label);
					get_neibour(get_superpixel(t).neibour_list.at(i).label);
					map.push_back(get_superpixel(t).neibour_list.at(i).label);
				}
			}
		}
	}
	if (c != 100) {
		update_depth(k, a, b, c);
	}
}

void ImgData::update_depth(int k, int a, int b, int c) {
	//cout << "find update ing for " << k << "...\n";
	//更新超像素对应像素的深度值
	get_superpixel(k).depth_average = ave(get_superpixel(a).depth_average, get_superpixel(b).depth_average, get_superpixel(c).depth_average);
	//cout << get_superpixel(a).depth_average << " " << get_superpixel(b).depth_average << " " << get_superpixel(c).depth_average << " " << get_superpixel(k).depth_average<<"\n";
	get_superpixel(k).depth_max = get_superpixel(k).depth_average;
	get_superpixel(k).depth_min = get_superpixel(k).depth_average;
	get_superpixel(k).depth_num = 0;
	float sum = 0;
	for (int i = 0; i < get_superpixel(k).pixels.size(); i++) {
		cv::Point p = get_superpixel(k).pixels.at(i);
		//int lf = get_superpixel(k).depth_min * 1000;
		//int rf = get_superpixel(k).depth_max * 1000;
		if (depth_mat.at<float>(p) > 1e-6) {
			depth_mat.at<float>(p) = (get_superpixel(k).depth_average+ depth_mat.at<float>(p))/2;
		}
		else depth_mat.at<float>(p) = get_superpixel(k).depth_average;
		get_superpixel(k).pixels_depth.at(i) = depth_mat.at<float>(p);
		get_superpixel(k).depth_num++;
		sum += depth_mat.at<float>(p);
		if (depth_mat.at<float>(p) > get_superpixel(k).depth_max)
			get_superpixel(k).depth_max = depth_mat.at<float>(p);
		if (depth_mat.at<float>(p) < get_superpixel(k).depth_min)
			get_superpixel(k).depth_min = depth_mat.at<float>(p);
	}
	get_superpixel(k).depth_average = sum / get_superpixel(k).depth_num;
}

bool SuperPixel::exist_source(int k) {
	for (int i = 0; i < source_list.size(); i++) {
		if (source_list.at(i).label == k)
			return true;
	}
	return false;
}

void ImgData::get_neibour(int k) {
	//得到超像素k的相邻超像素列表
	if (get_superpixel(k).neibour_list.size() > 0)
		return;
	double xx[] = { -1,1,0,0 };
	double yy[] = { 0,0,-1,1 };
	for (int i = 0; i < get_superpixel(k).contour.size(); i++) {
		cv::Point p = get_superpixel(k).contour.at(i);
		double x = p.x;
		double y = p.y;
		for (int j = 0; j < 4; j++) {
			if (x + xx[j] < 0 || x + xx[j] == WIDTH)
				continue;
			if (y + yy[j] < 0 || y + yy[j] == HEIGHT)
				continue;
			int l = sp_label.at<int>(Point(x+xx[j], y+yy[j]));
			if (l == k)
				continue;
			bool exist = false;
			if (get_superpixel(k).neibour_list.size() > 0) {
				for (int h = 0; h < get_superpixel(k).neibour_list.size(); h++){
					if (get_superpixel(k).neibour_list.at(h).label == l){
						exist = true;
						break;
					}
				}
			}
			if (!exist) {
				sp_source temp;
				temp.chi = compareHist(get_superpixel(k).hist, get_superpixel(l).hist, CV_COMP_CHISQR);
				temp.label = l;
				get_superpixel(k).neibour_list.push_back(temp);
			}
		}
	}
}

float ImgData::find_depth(Point& p_begin)
{
	//从p_begin点出发寻找最近的有深度的点 
	Mat visited(HEIGHT, WIDTH, CV_8UC1);
	queue<Point> queue_p;
	queue_p.push(p_begin);
	while (true)
	{
		Point p_next = queue_p.front();
		queue_p.pop();
		if (!check_range(p_next))
			continue;
		if (visited.at<int>(p_next) != 0)
			continue;

		visited.at<int>(p_next) = 1;	//记录该节点已访问
		float depth = get_pixel_depth(p_next);

		if (!is_zero(depth))
		{
			p_begin = p_next;		//更新输入顶点的值
			cout << "find depth point: " << p_next << " depth = " << depth << endl;
			return depth;
		}
		//把相邻的顶点加入队列
		queue_p.push(p_next + Point(-1, 0));
		queue_p.push(p_next + Point(1, 0));
		queue_p.push(p_next + Point(0, -1));
		queue_p.push(p_next + Point(0, 1));
	}
}

void ImgData::calc_world_center()     //全部像素点的世界坐标中心
{
	Mat temp = Mat::zeros(3, 1, CV_32F);
	int depth_num = 0;
	for (int x = 0; x < WIDTH; x++)
	{
		for (int y = 0; y < HEIGHT; y++)
		{
			float depth = get_pixel_depth(Point(x, y));
			if (depth > 1e-6)
			{
				temp += cam.get_world_pos(Point(x, y), depth);
				depth_num++;
			}
		}
	}
	world_center = temp / depth_num;
}

SuperPixel& ImgData::get_superpixel(int i) 
{ 
	return data[i]; 
}

float& ImgData::get_pixel_depth(Point& point)
{
	return depth_mat.at<float>(point);
}

void shape_preserve_wrap(ImgData& imgdata, Camera& novel_cam, Mat& output_img, int thread_rank)
{
	cout << "--thread--" << thread_rank << "--begin shape_preserve_wrap..." << endl;
	clock_t start;
	clock_t end;
	start = clock();

	output_img = Mat::zeros(HEIGHT, WIDTH, CV_8UC3);
	Mat wrap_img_depth = Mat::zeros(HEIGHT, WIDTH, CV_32F);			//记录wrap后img的深度图
	Mat reproject_mat, reproject_vec;
	imgdata.cam.fill_reprojection(novel_cam, reproject_mat, reproject_vec);

	// 计算每个超像素在新视点下的深度
	vector<float> depth_dict(imgdata.sp_num);
	for (int i = 0; i < imgdata.sp_num; i++)
	{
		SuperPixel& superpixel = imgdata.get_superpixel(i);
		if (!superpixel.have_depth())
			continue;                      //若原本无深度值则忽略
		Mat temp_mat = novel_cam.cam_pos - imgdata.cam.get_world_pos(superpixel.center, superpixel.depth_average);   //人为计算深度值
		depth_dict[i] = sqrt(temp_mat.dot(temp_mat));
	}


	clock_t t1 = 0;


	// 逐个超像素进行wrap
	for (int i = 0; i < imgdata.sp_num; i++)
	{
		SuperPixel& superpixel = imgdata.get_superpixel(i);
		if (!superpixel.have_depth())
			continue;
		vector<Point2f> triangle;
		minEnclosingTriangle(superpixel.pixels, triangle);			// 计算最小外接三角形

																	//calculate ep
        Eigen::Matrix<float, 6, 6> ep_mat_eigen = Eigen::Matrix<float, 6, 6>::Zero();
        Eigen::Matrix<float, 6, 1 >ep_vec_eigen = Eigen::Matrix<float, 6, 1>::Zero();

        Eigen::MatrixXf A_mat = Eigen::MatrixXf::Zero(superpixel.pixel_num * 2, 6);
        Eigen::MatrixXf b_mat = Eigen::MatrixXf::Zero(superpixel.pixel_num * 2, 1);

        vector<float> coefficient(3);
		{
			for (int j = 0; j < superpixel.pixel_num; j++)
			{
				Point& origin_point = superpixel.get_pixel(j);
				float point_depth = superpixel.pixels_depth[j];
				// 检验是否有深度
				if (point_depth < 1e-6)
					continue;

				// 插值到外接三角形里，得到用三个定点表示的参数

				tri_interpolation(triangle, origin_point, coefficient);

				// 计算在新视点下的像素坐标	
				Point destination_point = cal_reprojection(origin_point, point_depth, reproject_mat, reproject_vec);

                for (int i = 0; i < 3; i++) A_mat(2 * j, i) = A_mat(2 * j + 1, i + 3) = coefficient[i % 3];
                b_mat(2 * j, 0) = destination_point.x;
                b_mat(2 * j + 1, 0) = destination_point.y;
			}
		}

        ep_mat_eigen = A_mat.transpose()*A_mat;
        ep_vec_eigen = A_mat.transpose()*b_mat;

		// 计算es_mat，衡量三角形的形变量
        Eigen::Matrix<float, 6, 6> es_mat_eigen = Eigen::Matrix<float, 6, 6>::Zero();
		{
			int j, k, l;
			for (int iter_time = 0; iter_time < 3; iter_time++)
			{
				switch (iter_time)
				{
				case 0:
					j = 0; k = 1; l = 2;
					break;
				case 1:
					j = 1; k = 2; l = 0;
					break;
				case 2:
					j = 2; k = 0; l = 1;
					break;
				default:
					break;
				}

				Point2f& pj = triangle[j]; float xj = pj.x, yj = pj.y;
				Point2f& pk = triangle[k]; float xk = pk.x, yk = pk.y;
				Point2f& pl = triangle[l]; float xl = pl.x, yl = pl.y;

                Eigen::Vector2f p1(xk, yk);
                Eigen::Vector2f p2(xj, yj);
                Eigen::Vector2f p3(xl, yl);

                Eigen::Matrix2f R90;
                R90 << 
                    0, 1,
                    -1, 0;

                Eigen::Matrix<float, 2, 6> A = Eigen::Matrix<float, 2, 6>::Zero();
                float a = (p3 - p1).dot(p2 - p1) / (p2 - p1).squaredNorm();
                float b = (p3 - p1).dot(R90*(p2 - p1)) / (p2 - p1).squaredNorm();

                //E=|p3-p1-a(p2-p1)-bR90(p2-p1)|^2
                Eigen::Matrix2f Aj, Ak, Al; 
                Ak <<
                    (-1 + a), b,
                    -b, (-1 + a);
                Aj <<
                    -a, -b,
                    b, -a;
                Al << 
                    1, 0,
                    0, 1;
                A.col(j) = Aj.col(0);
                A.col(j + 3) = Aj.col(1);
                A.col(k) = Ak.col(0);
                A.col(k + 3) = Ak.col(1);
                A.col(l) = Al.col(0);
                A.col(l + 3) = Al.col(1);

                Eigen::Matrix<float, 6, 6 >ATA = A.transpose()*A;
                es_mat_eigen += ATA;
			}
		}

		// 求逆矩阵，计算在新视点下的外接三角形
		float es_weight = 1;

        Eigen::Matrix<float, 6, 6> temp_mat_eigen = ep_mat_eigen + es_mat_eigen*es_weight;
        if (temp_mat_eigen.determinant() < 1e-6) continue;
        Eigen::LDLT<Eigen::MatrixXf> ldlt(temp_mat_eigen);
        Eigen::VectorXf x = ldlt.solve(ep_vec_eigen);

		vector<Point2f> novel_triangle;
		novel_triangle.resize(3);
		novel_triangle[0] = Point2f(x(0), x(3));
		novel_triangle[1] = Point2f(x(1), x(4));
		novel_triangle[2] = Point2f(x(2), x(5));

		//如果面积之比大于4，则跳过
		float origin_area = calc_triangle_area(triangle);
		float new_area = calc_triangle_area(novel_triangle);
		if (new_area / origin_area > 4)
			continue;


		// 把原来超像素的轮廓用三角形插值投影到新视点下
        vector<Point> novel_contour(superpixel.contour.size());

        vector<Point> novel_points;
		for (int j = 0; j < superpixel.contour.size(); j++)
		{
			Point& origin_point = superpixel.contour[j];
			tri_interpolation(triangle, origin_point, coefficient);
            novel_contour[j] = (inv_tri_interpolation(novel_triangle, coefficient));
		}

		// 用投影后的轮廓得到投影后的超像素区域
		contour_to_set(novel_contour, novel_points);

		for (int j = 0; j < novel_points.size(); j++)
		{
			Point& novel_point = novel_points[j];
			vector<float> coefficient;
			tri_interpolation(novel_triangle, novel_point, coefficient);
			Point reproject_point = inv_tri_interpolation(triangle, coefficient);
			if (check_range(reproject_point) && check_range(novel_point))
			{
				float& before_depth = wrap_img_depth.at<float>(novel_point);
				if (abs(before_depth - 0) < 1e-6 || depth_dict[i] < before_depth)
				{
					before_depth = depth_dict[i];
					output_img.at<Vec3b>(novel_point) = imgdata.origin_img.at<Vec3b>(reproject_point);
				}
			}
		}
	}
	end = clock();
	cout << "--thread--" << thread_rank << "--end using time:..." << (end - start) << "ms" << endl;
}

void mix_pic(vector<ImgData>& imgdata_vec, Camera& now_cam, vector<int>& img_id, Mat& output_img)
{
	cout << "--begin generate pic..." << endl;
	clock_t start;
	clock_t end;
	start = clock();
	output_img = Mat::zeros(HEIGHT, WIDTH, CV_8UC3);
	vector<Mat> wrap_img(img_id.size());
	vector<thread> threads(img_id.size());
	for (int i = 0; i < img_id.size(); i++)
	{
		//对四个最近的照片分别wrap到同一个视点
		threads[i] = thread(shape_preserve_wrap, ref(imgdata_vec[img_id[i]]), ref(now_cam), ref(wrap_img[i]), i);
	}
	//等待所有线程执行完毕
	for (int i = 0; i < threads.size(); i++)
	{
		threads[i].join();
	}
	//图像融合
	for (int x = 0; x < WIDTH; x++)
	{
		for (int y = 0; y < HEIGHT; y++)
		{
			Point point(x, y);
			for (int i = 0; i < img_id.size(); i++)
			{
				//按照片远近优先级赋值
				if (wrap_img[i].at<Vec3b>(point) != Vec3b{ 0,0,0 })
				{
					output_img.at<Vec3b>(point) = wrap_img[i].at<Vec3b>(point);
					break;		//一旦找到一张wrap后图像有图像信息，则赋值完跳出循环
				}
			}

		}
	}

	end = clock();
	cout << "--OK using time:" << (end - start) << "ms" << endl;

}
