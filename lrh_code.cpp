#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>


cv::Mat fast_guided_filter_8(const cv::Mat& I_org, int r, float eps, int s)
{
	cv::Mat I, _I;
	I_org.convertTo(_I, CV_32FC1);
	resize(_I, I, cv::Size(), 1.0 / s, 1.0 / s, 1);

	// int hei = I.rows;
	// int wid = I.cols;
	r = std::max(3, (2 * r + 1) / s);

	cv::Mat mean_I;
	cv::boxFilter(I, mean_I, CV_32FC1, cv::Size(r, r)); //I做3x3均值滤波

	cv::Mat mean_II;
	cv::boxFilter(I.mul(I), mean_II, CV_32FC1, cv::Size(r, r)); //I平方做3x3均值滤波

	cv::Mat var_I = mean_II - mean_I.mul(mean_I);

	cv::Mat a = var_I / (var_I + eps);

	cv::Mat b = mean_I - a.mul(mean_I);

	cv::Mat rmean_a;
	resize(a, rmean_a, cv::Size(I_org.cols, I_org.rows), 1);

	cv::Mat rmean_b;
	resize(b, rmean_b, cv::Size(I_org.cols, I_org.rows), 1);

	cv::Mat q;
	cv::add(rmean_a.mul(_I), rmean_b, q);
	q.convertTo(q, CV_8UC1);
	return q;
}

void dde8(const cv::Mat& src, cv::Mat& dst, float sharpness)
{
	// 8位细节增强
	cv::Mat base8 = fast_guided_filter_8(src, 1, 100, 1);
	cv::Mat detail8;
	cv::subtract(src, base8, detail8, cv::noArray(), CV_8UC1);
	cv::multiply(detail8, 1, detail8, sharpness, CV_8UC1);
	cv::add(src, detail8, dst, cv::noArray(), CV_8UC1);
}


int main(int argc, char** argv)
{	
	//cv::Mat gray_img;
	cv::Mat out_img;
	const char* img_path = argv[1];
	int sharpness = atoi(argv[2]);

	cv::Mat img = cv::imread(img_path, 0);
	if(img.empty())
	{
		std::cout << "read image failed" << std::endl;
		return -1;
	}

	//cv::cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);

	dde8(img, out_img, sharpness);
	cv::imwrite("./test.bmp", out_img);
}
