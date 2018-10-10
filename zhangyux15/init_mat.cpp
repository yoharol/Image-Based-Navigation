#include "init_mat.h"

#include <opencv/cv.h>

using namespace cv;

Mat init_mat(int rows, int cols, int type, int a[])
{
	return Mat(rows, cols, type, a);
}