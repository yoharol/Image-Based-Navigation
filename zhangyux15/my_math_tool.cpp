#include "my_math_tool.h"
#include "global_var.h"

#include <cmath>
#include <opencv/cv.h>

#include <opencv2/ximgproc.hpp>
#include <opencv2/ximgproc/slic.hpp>

using namespace std; 
using namespace cv;

bool is_zero(float x)
{
	return abs(x) < 1e-6;
}

bool check_range(Point& point)
{
	return ((point.x >= 0) && (point.x < WIDTH) && (point.y >= 0) && (point.y < HEIGHT));
}

void split_string(const std::string& s, std::vector<std::string>& v, const std::string& c)
{
	std::string::size_type pos1, pos2;
	pos2 = s.find(c);
	pos1 = 0;
	while (std::string::npos != pos2)
	{
		v.push_back(s.substr(pos1, pos2 - pos1));

		pos1 = pos2 + c.size();
		pos2 = s.find(c, pos1);
	}
	if (pos1 != s.length())
		v.push_back(s.substr(pos1));
}

void calc_sp(Mat &origin_img, Mat &output_sp_label, Mat &output_sp_contour, int &output_sp_num)
{
	cout << "--create superpixels..." << std::flush;
	clock_t start;
	clock_t end;
	start = clock();

	Mat frame, labels, mask;
	frame = origin_img.clone();				// 拷贝一份原图
	Ptr<cv::ximgproc::SuperpixelSLIC> slic =
		cv::ximgproc::createSuperpixelSLIC
		(frame, ximgproc::SLIC, 10, 10.0f);	// 生成超像素
	slic->iterate();
	slic->enforceLabelConnectivity();
	slic->getLabels(output_sp_label);
	slic->getLabelContourMask(output_sp_contour, true);
	output_sp_num = slic->getNumberOfSuperpixels();	// 获取超像素数量

	/*
	frame.setTo(Scalar(255, 255, 255), mask);
	imwrite(filename, frame);
	*/
	end = clock();
	cout << "OK using time:" << (end - start) << "ms" << endl;
}

void tri_interpolation(vector<Point2f>& triangle, Point& point, vector<float>& coefficient)
{
	coefficient.resize(3);
	float px = point.x; float py = point.y;
	float& p0x = triangle[0].x; float& p0y = triangle[0].y;
	float& p1x = triangle[1].x; float& p1y = triangle[1].y;
	float& p2x = triangle[2].x; float& p2y = triangle[2].y;

	float divisor = (p0x*p1y - p1x*p0y - p0x*p2y + p2x*p0y + p1x*p2y - p2x*p1y);

	coefficient[1] = -(p0x*p2y - p2x*p0y + p0y*px - p2y*px - p0x*py + p2x*py) / divisor;
	coefficient[2] = (p0x*p1y - p1x*p0y + p0y*px - p1y*px - p0x*py + p1x*py) / divisor;
	coefficient[0] = 1 - coefficient[1] - coefficient[2];
}

Point inv_tri_interpolation(vector<Point2f>& triangle, vector<float>& coefficient)
{
	return Point(coefficient[0] * triangle[0] + coefficient[1] * triangle[1] +
		coefficient[2] * triangle[2]);
}


float point_distance(cv::Point2f p1, cv::Point2f p2)
{
	return(sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2)));
}


void contour_to_set(vector<Point>& contour, vector<Point>& points_set)
{
	approxPolyDP(contour, contour, 0, true);
	points_set.resize(0);
	CvRect rect_bound = boundingRect(contour);
	Point zero_point = Point(rect_bound.x, rect_bound.y);
	// 新建一个超像素外接矩形大小的Mat
	Mat draft = Mat::zeros(rect_bound.height, rect_bound.width, CV_8U);

	vector<Point> copy_contour;
	for (int i = 0; i < contour.size(); i++)
	{
		copy_contour.push_back(contour[i] - zero_point);
	}

	fillConvexPoly(draft, copy_contour, 1);

	for (int x = 0; x < rect_bound.width; x++)
	{
		for (int y = 0; y< rect_bound.height; y++)
		{
			if (draft.at<uchar>(Point(x, y)) == 1)
			{
				Point point(rect_bound.x + x, rect_bound.y + y);
				if (check_range(point))
				{
					points_set.push_back(point);
				}
			}
				
		}
	}
}


float calc_triangle_area(vector<Point2f>& triangle)
{
	Point2f p1 = triangle[1] - triangle[0];
	Point2f p2 = triangle[2] - triangle[0];
	return abs(p1.cross(p2));
}


vector<float> fit_plane(vector<Mat> point_set)
{
	vector<float> result(4);
	CvMat*points = cvCreateMat(point_set.size(), 3, CV_32FC1);//定义用来存储需要拟合点的矩阵   
	for (int i = 0; i < point_set.size(); ++i)
	{
		points->data.fl[i * 3 + 0] = point_set[i].at<float>(0, 0);//矩阵的值进行初始化   X的坐标值  
		points->data.fl[i * 3 + 1] = point_set[i].at<float>(1, 0);//  Y的坐标值  
		points->data.fl[i * 3 + 2] = point_set[i].at<float>(2, 0);//  Z的坐标值

	}
	int nrows = points->rows;
	int ncols = points->cols;
	int type = points->type;
	CvMat* centroid = cvCreateMat(1, ncols, type);
	cvSet(centroid, cvScalar(0));
	for (int c = 0; c<ncols; c++) {
		for (int r = 0; r < nrows; r++)
		{
			centroid->data.fl[c] += points->data.fl[ncols*r + c];
		}
		centroid->data.fl[c] /= nrows;
	}
	// Subtract geometric centroid from each point.  
	CvMat* points2 = cvCreateMat(nrows, ncols, type);
	for (int r = 0; r<nrows; r++)
		for (int c = 0; c<ncols; c++)
			points2->data.fl[ncols*r + c] = points->data.fl[ncols*r + c] - centroid->data.fl[c];
	// Evaluate SVD of covariance matrix.  
	CvMat* A = cvCreateMat(ncols, ncols, type);
	CvMat* W = cvCreateMat(ncols, ncols, type);
	CvMat* V = cvCreateMat(ncols, ncols, type);
	cvGEMM(points2, points, 1, NULL, 0, A, CV_GEMM_A_T);
	cvSVD(A, W, NULL, V, CV_SVD_V_T);
	// Assign plane coefficients by singular vector corresponding to smallest singular value.  
	result[ncols] = 0;
	for (int c = 0; c<ncols; c++) {
		result[c] = V->data.fl[ncols*(ncols - 1) + c];
		result[ncols] += result[c] * centroid->data.fl[c];
	}
	// Release allocated resources.  
	cvReleaseMat(&centroid);
	cvReleaseMat(&points2);
	cvReleaseMat(&A);
	cvReleaseMat(&W);
	cvReleaseMat(&V);

	return result;
}