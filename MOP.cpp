#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>

using namespace cv;
using namespace std;

Mat img, img2, img3, KMO1, KMO2, KMO3, KMO, dst, image, Mask, gris, clear;
Mat KMO_norm, KMO_abs;

int w = 50;
int W_slider = 0;
int minDistance = 10;
int QL = 50;
double qualityLevel = (QL)*0.01;
double W;
const double W_max = 20;
char* source_window = "Multi-Objetive Parameterized Interest Point Detector (MOP)";

/// Function headers
void nonMaximaSuppression(const Mat& src, Mat& dst1, const int sz, double qualityLevel, const Mat mask1);
void MOP(int, void*);


/// Multi-Objetive Parameterized Interest Point Detector (MOP)
int main()
{
	img = imread("lena.jpg", CV_LOAD_IMAGE_COLOR);

	gris = img.clone();
	cvtColor(gris, gris, CV_BGR2GRAY);
	clear = gris.clone();

	img.convertTo(img, CV_32F);
	cvtColor(img, image, CV_BGR2GRAY);

	namedWindow(source_window, CV_WINDOW_AUTOSIZE);
	createTrackbar("W:", source_window, &W_slider, W_max, MOP);
	imshow(source_window, gris);

	MOP(0, 0);

	waitKey(0);
	return 0;
}

/// MOP FUNCTION
void MOP(int, void*)
{
	gris = clear.clone();

	W = (double)W_slider / W_max;
	cout << W << endl;

	// KMO_1
	pow(image, 2, img2);                                // I^2
	GaussianBlur(img2, img2, Size(9, 9), 1.0, 1.0);     //G1*I ^ 2
	log(img2, img2);                                    // log(G1*I^2)
	GaussianBlur(img2, KMO1, Size(9, 9), 1.0, 1.0);     // G1*log(G1*I^2)

    // KMO_2
	GaussianBlur(image, img3, Size(9, 9), 1.0, 1.0);    // G1*I
	absdiff(img3, image, img3);                         // |(G1*I)-I|
	GaussianBlur(img3, KMO2, Size(9, 9), 2.0, 2.0);     // G2 * |(G1*I)-I|
	KMO2 = (W * KMO2);

	// KMO_3
	//GaussianBlur(image, KMO3, Size(9, 9), 1.0, 1.0);// G1 * I
	//divide(KMO3, image, KMO3); // (G1 * I)/I

	/// KMO
	KMO = KMO1 + KMO2;                                  //  KMO_1 + (W x KMO_2)
    // KMO = KMO1 + KMO2 + KMO3;                        //KMO_1 + (W x KMO_2) + KMO_3
    KMO = abs(KMO);
	pow(KMO, 2, KMO);                                   //[KMO_1 + (W x KMO_2)]^2
	GaussianBlur(KMO, KMO, Size(9, 9), 2.0, 2.0);       // G2 * [KMO_1 + (W x KMO_2)]^2

	normalize(KMO, KMO_norm, 0, 255, NORM_MINMAX);
	convertScaleAbs(KMO_norm, KMO_abs);
	imshow("KMO", KMO_abs);

	if (Mask.empty())
	{
		Mask = Mat::zeros(KMO.size(), CV_8UC1);
		Mask(Rect(minDistance, minDistance, (KMO.cols) - (2 * minDistance), (KMO.rows) - (2 * minDistance))) = 1;
	}

	//Non Maxima Suppresion application
	nonMaximaSuppression(KMO, dst, minDistance, qualityLevel, Mask);

	imshow("Mask", dst);

	for (int j = 0; j < dst.rows; j++)
	{
		for (int i = 0; i < dst.cols; i++)
		{
			if ((int)dst.at<uchar>(j, i) > 200)
			{
				circle(gris, Point(i, j), 4, Scalar(255, 255, 255), 2, 8, 0);
			}
		}
	}

	imshow(source_window, gris);
}

/// NMS
void nonMaximaSuppression(const Mat& src, cv::Mat& dst1, const int sz, double qualityLevel, const cv::Mat mask1)
{

	double minStrength;
	double maxStrength;
	int threshold1;

	minMaxLoc(src, &minStrength, &maxStrength);
	threshold1 = qualityLevel*maxStrength;
	threshold(src, src, threshold1, 255, 3);

	const int M = src.rows;
	const int N = src.cols;
	const bool masked = !mask1.empty();
	Mat block = 255 * Mat_<uint8_t>::ones(Size(2 * sz + 1, 2 * sz + 1));
	dst1 = Mat_<uint8_t>::zeros(src.size());

	for (int m = 0; m < M; m += sz + 1)
	{
		for (int n = 0; n < N; n += sz + 1)
		{
			Point  ijmax;
			double vcmax, vnmax;

			Range ic(m, min(m + sz + 1, M));
			Range jc(n, min(n + sz + 1, N));
			minMaxLoc(src(ic, jc), NULL, &vcmax, NULL, &ijmax, masked ? mask1(ic, jc) : noArray());
			Point cc = ijmax + Point(jc.start, ic.start);
			Range in(max(cc.y - sz, 0), min(cc.y + sz + 1, M));
			Range jn(max(cc.x - sz, 0), min(cc.x + sz + 1, N));

			Mat_<uint8_t> blockmask;
			block(Range(0, in.size()), Range(0, jn.size())).copyTo(blockmask);
			Range iis(ic.start - in.start, min(ic.start - in.start + sz + 1, in.size()));
			Range jis(jc.start - jn.start, min(jc.start - jn.start + sz + 1, jn.size()));
			blockmask(iis, jis) = Mat_<uint8_t>::zeros(Size(jis.size(), iis.size()));
			minMaxLoc(src(in, jn), NULL, &vnmax, NULL, &ijmax, masked ? mask1(in, jn).mul(blockmask) : blockmask);

			if (vcmax > vnmax)
                {

				dst1.at<uint8_t>(cc.y, cc.x) = 255;

                }
		}
	}
}

