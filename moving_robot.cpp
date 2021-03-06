#include <iostream>    
#include<opencv2/highgui/highgui.hpp>  
#include<opencv2/imgproc/imgproc.hpp>  
#include<opencv2\core.hpp> 
#include <opencv2/imgproc/types_c.h>
#include<fstream>
#include<string>
#include<assert.h>

using namespace cv;
using namespace std;
//
//vector<Point> points;
//Point center;
////绘制贝塞尔曲线  
//Point pointAdd(Point p, Point q) {
//	p.x += q.x;        p.y += q.y;
//	return p;
//}
//
//Point pointTimes(float c, Point p) {
//	p.x *= c;    p.y *= c;
//	return p;
//}
//
//Point Bernstein(float u, Point qi, Point mid, Point mo)
//{
//	Point a, b, c, r;
//
//	a = pointTimes(pow(u, 2), mo);
//	b = pointTimes(pow((1 - u), 2), qi);
//	c = pointTimes(2 * u*(1 - u), mid);
//
//	r = pointAdd(pointAdd(a, b), c);
//
//	return r;
//}
//
//int main(int argc, char** argv)
//{
//	VideoCapture cap(1);//读取USB摄像头  
//	if (!cap.isOpened())
//		return -1;
//
//	int iLowH = 0;
//	int iHighH = 5;
//	int iLowS = 45;
//	int iHighS = 255;
//	int iLowV = 45;
//	int iHighV = 255;
//	int nGaussianBlurValue = 3;
//
//	//采取颜色识别方法，利用滑条选色，参考HSV对应的颜色，获取目标物体  
//
//	namedWindow("Control");
//	cvCreateTrackbar("LowH", "Control", &iLowH, 179); //Hue (0 - 179)    
//	cvCreateTrackbar("HighH", "Control", &iHighH, 179);
//
//	cvCreateTrackbar("LowS", "Control", &iLowS, 255); //Saturation (0 - 255)    
//	cvCreateTrackbar("HighS", "Control", &iHighS, 255);
//
//	cvCreateTrackbar("LowV", "Control", &iLowV, 255); //Value (0 - 255)    
//	cvCreateTrackbar("HighV", "Control", &iHighV, 255);
//
//	while (true)
//	{
//		Mat imgOriginal;
//		cap >> imgOriginal;
//		//高斯滤波  
//		GaussianBlur(imgOriginal, imgOriginal, Size(nGaussianBlurValue*+1, nGaussianBlurValue * 2 + 1), 0, 0);
//
//		Mat imgHSV;
//		vector<Mat> hsvSplit;
//		cvtColor(imgOriginal, imgHSV, COLOR_BGR2HSV); //转换颜色空间  
//
//		Mat element1 = getStructuringElement(MORPH_RECT, Size(5, 5));//获取结构元素  
//		morphologyEx(imgHSV, imgHSV, MORPH_OPEN, element1);//开操作  
//		morphologyEx(imgHSV, imgHSV, MORPH_CLOSE, element1);//闭操作  
//
//
//		split(imgHSV, hsvSplit);//HSV图像分离  
//		equalizeHist(hsvSplit[2], hsvSplit[2]);//直方图均衡化  
//		merge(hsvSplit, imgHSV);//HSV图像聚合  
//		Mat imgThresholded;
//		//根据颜色选取目标物体  
//		inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded);
//
//
//		Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));//获取结构元素  
//		morphologyEx(imgThresholded, imgThresholded, MORPH_OPEN, element);//开操作  
//		morphologyEx(imgThresholded, imgThresholded, MORPH_CLOSE, element);//闭操作
//		morphologyEx(imgThresholded, imgThresholded, MORPH_ELLIPSE, element);//膨胀操作
//		vector<vector<Point>> contours;
//		vector<Vec4i> hierarcy;
//		findContours(imgThresholded, contours, hierarcy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);//查找轮廓
//																								 //drawContours(imgOriginal, contours, -1, Scalar(0, 255, 0), 2);//绘制轮廓
//		vector<RotatedRect> box(contours.size());
//		for (int i = 0; i < contours.size(); i++)
//		{
//			box[i] = fitEllipse(Mat(contours[i]));
//			center = box[i].center;
//			points.push_back(center);
//			//circle(imgOriginal, center, 3, Scalar(0, 255, 0));//绘制目标物体质点
//			//ellipse(imgOriginal, box[i], Scalar(0, 255, 0));//绘制拟合椭圆
//			for (int j = 2; j < points.size(); j += 2)
//			{
//				Point pre, last, mid;
//				pre = points[j - 2];
//				mid = points[j - 1];
//				last = points[j];
//				Point pt_pre = points[j - 2];
//				Point pt_now;
//				//绘制贝塞尔曲线,一小段一小段的直线就能组合成曲线
//				for (int k = 0; k <= 10; k++)
//				{
//					float u = (float)k / 10;
//					Point new_point = Bernstein(u, pre, mid, last);
//					pt_now.x = (int)new_point.x;
//					pt_now.y = (int)new_point.y;
//					line(imgOriginal, pt_pre, pt_now, Scalar(0, 255, 0), 2, CV_AA, 0);//绘制直线
//					pt_pre = pt_now;
//				}
//			}
//		}
//		imshow("Thresholded Image", imgThresholded); //显示处理图像
//		imshow("Original", imgOriginal); //显示最终图像 
//		char key = (char)waitKey(300);
//		if (key == 27)
//			break;
//	}
//	return 0;
//}
//


//#include <opencv2/opencv.hpp>
//using namespace std;
//using namespace cv;
//
//int main(int argc, const char** argv)
//{
//	Mat img = imread("C:\\Users\\善锐刘\\xt.jpg", IMREAD_GRAYSCALE);
//	//threshold(img, img, 200, 255, CV_THRESH_BINARY); // to delete some noise
//	Mat labels;
//	connectedComponents(img, labels, 8, CV_16U);//连通域提取
//	Mat result(img.size(), CV_32FC1, Scalar::all(0));
//	for (int i = 0; i <= 1; i++)
//	{
//		Mat mask1 = labels == 1 + i;//提取标签信息，当满足条件返回255，不满足返回0
//		Mat mask2 = labels == 1 + (1 - i);
//		Mat masknot;
//		bitwise_not(mask1, masknot);
//		Mat dist;
//		distanceTransform(masknot, dist, DIST_L2, 5, CV_8U);
//		dist.copyTo(result, mask2); //核心一句，直接提取距离信息
//	}
//	FileStorage fs("distCtr.yml", FileStorage::WRITE);
//	fs << "Image" << result;
//	fs.release();
//	return 0;
//}


//  线条轨迹二值化


//
//#include <iostream>
//#include <opencv2/opencv.hpp>
//#include <fstream>
//using namespace std;
//using namespace cv;
//
//int main()
//{
//	Mat src = imread("C:/Users/善锐刘/Pictures/s1.jpg", 0);
//	//imshow("src", src);
//
//	Mat dst;
//	threshold(src, dst, 100, 255,1);		//二值化
//	/*
//enum ThresholdTypes {
//    THRESH_BINARY     = 0,
//    THRESH_BINARY_INV = 1,
//    THRESH_TRUNC      = 2,
//    THRESH_TOZERO     = 3,
//    THRESH_TOZERO_INV = 4,
//    THRESH_MASK       = 7,
//    THRESH_OTSU       = 8,
//    THRESH_TRIANGLE   = 16
//};*/
//
//														    //imshow("dst", dst);
//
//	int nRows = dst.rows;
//	int nCols = dst.cols;
//
//	ofstream fout("data.txt");
//
//	//按列扫描，求像素和，由于是二值后的图片，没有线条时，该列的像素和为0；扫描到线条时像素大于0
//	for (int w = 0; w < nCols; w++)
//	{
//		int sum = 0;
//
//		for (int h = 0; h < nRows; h++)
//		{
//			uchar *pRow = dst.ptr<uchar>(h, w); //该列中每个像素的地址
//			sum += (int)(*pRow);
//
//			if (sum > 0)  //到达了线条的上侧，像素和大于0
//			{
//				cout << "找到了线条点,";   //从上往下找，由于线条很细，目前只判断上边界。
//				cout << "坐标如下： X = " << w << ", Y = " << h << endl;
//				fout << "坐标如下： X = " << w << ", Y = " << h << endl; //控制台会丢失数据，存到文本不会丢失
//				sum = 0;
//				break;
//			}
//		}
//	}
//
//	waitKey();
//	cout << endl;
//	system("pause");
//	return 0;
//}




   //实时采集图像代码：




//#include "core/core.hpp"      
//#include "highgui/highgui.hpp"      
//#include "imgproc/imgproc.hpp"  
//#include "video/tracking.hpp"  
//#include<iostream>      
//
//using namespace cv;
//using namespace std;
//
//Mat image;
//Mat rectImage;
//Mat imageCopy; //绘制矩形框时用来拷贝原图的图像    
//bool leftButtonDownFlag = false; //左键单击后视频暂停播放的标志位    
//Point originalPoint; //矩形框起点    
//Point processPoint; //矩形框终点    
//
//Mat targetImageHSV;
//int histSize = 200;
//float histR[] = { 0,255 };
//const float *histRange = histR;
//int channels[] = { 0,1 };
//Mat dstHist;
//Rect rect;
//vector<Point> pt; //保存目标轨迹  
//void onMouse(int event, int x, int y, int flags, void* ustc);   //鼠标回调函数 
//
//
//int main(int argc, char*argv[])
//{
//	//VideoCapture video("C:/Users/善锐刘/Pictures/tt.mp4 ");
//	VideoCapture video(0);
//	while(true){
//		//double fps = frame.get(CAP_PROP_FPS); //获取视频帧率    
//		//double pauseTime = 1000 / fps; //两幅画面中间间隔   
//		double fps = video.get(CAP_PROP_FPS);
//		double pauseTime = 1000 / fps;
//		namedWindow("跟踪木头人", 0);
//		setMouseCallback("跟踪木头人", onMouse);
//		while (true)
//		{
//			if (!leftButtonDownFlag) //判定鼠标左键没有按下，采取播放视频，否则暂停    
//			{
//				video >> image;    // 读取图像帧至image
//			}
//			if (!image.data || waitKey(pauseTime) == 27)  //图像为空或Esc键按下退出播放    
//			{
//				break;
//			}
//			if (originalPoint != processPoint && !leftButtonDownFlag)
//			{
//				Mat imageHSV;
//				Mat calcBackImage;
//				cvtColor(image, imageHSV, CV_RGB2HSV);
//				calcBackProject(&imageHSV, 2, channels, dstHist, calcBackImage, &histRange);  //反向投影  
//				TermCriteria criteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 1000, 0.001);
//				CamShift(calcBackImage, rect, criteria);
//				Mat imageROI = imageHSV(rect);   //更新模板             
//				targetImageHSV = imageHSV(rect);
//				calcHist(&imageROI, 2, channels, Mat(), dstHist, 1, &histSize, &histRange);
//				normalize(dstHist, dstHist, 0.0, 1.0, NORM_MINMAX);   //归一化  
//				rectangle(image, rect, Scalar(255, 0, 0), 3);    //目标绘制    
//				pt.push_back(Point(rect.x + rect.width / 2, rect.y + rect.height / 2));
//				for (int i = 0; i<pt.size() - 1; i++)
//				{
//					line(image, pt[i], pt[i + 1], Scalar(0, 255, 0), 2.5);
//				}
//			}
//			imshow("跟踪木头人", image);
//			waitKey(100);
//		}
//	}
//	
//	return 0;
//}

//*******************************************************************//      
//鼠标回调函数 

//void onMouse(int event, int x, int y, int flags, void *ustc)
//{
//	if ( event == EVENT_LBUTTONDOWN )
//	{
//		leftButtonDownFlag = true; //标志位    
//		originalPoint = Point(x, y);  //设置左键按下点的矩形起点    
//		processPoint = originalPoint;
//	}
//	if ( event == EVENT_MOUSEMOVE && leftButtonDownFlag )
//	{
//		imageCopy = image.clone();
//		processPoint = Point(x, y);
//		if (originalPoint != processPoint)
//		{
//			//在复制的图像上绘制矩形    
//			rectangle(imageCopy, originalPoint, processPoint, Scalar(255, 0, 0), 2);
//		}
//		imshow("跟踪木头人", imageCopy);
//	}
//	if (event == EVENT_LBUTTONUP)
//	{
//		leftButtonDownFlag = false;
//		rect = Rect(originalPoint, processPoint);
//		rectImage = image(rect); //子图像显示    
// 		imshow("Sub Image", rectImage);
//		cvtColor(rectImage, targetImageHSV, CV_RGB2HSV);
//		imshow("targetImageHSV", targetImageHSV);
//		calcHist(&targetImageHSV, 2, channels, Mat(), dstHist, 1, &histSize, &histRange, true, false);
//		normalize(dstHist, dstHist, 0, 255, 32);
//		imshow("dstHist", dstHist);
//	}
//}


//// 提取指定颜色的所有像素

#include<iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
using namespace cv;

class ColorDetector
{
private:
	//最小可接受距离
	int minDist;
	//目标色
	cv::Vec3b target;
	//结果图像
	cv::Mat result;

	//计算与目标颜色的距离
	int getDistance(cv::Vec3b color)
	{
		return abs(color[0] - target[0]) + abs(color[1] - target[1]) + abs(color[2] - target[2]);
	}
public:
	//空构造函数
	ColorDetector() :minDist(100)
	{
		//初始化默认参数
		target[0] = target[1] = target[2] = 0;
	}
	void setColorDistanceThreshold(int distance);
	int getColorDistanceThreshold() const;
	void setTargetColor(unsigned char red, unsigned char green, unsigned char blue);
	void setTargetColor(cv::Vec3b color);
	cv::Vec3b getTargetColor() const;
	cv::Mat ColorDetector::process(const cv::Mat &image);
};
//设置色彩距离阈值，阈值必须是正的，否则设为0
void ColorDetector::setColorDistanceThreshold(int distance)
{
	if (distance < 0)
		distance = 0;
	minDist = distance;
}
//获取色彩距离阈值
int ColorDetector::getColorDistanceThreshold() const
{
	return minDist;
}
//设置需检测的颜色
void ColorDetector::setTargetColor(unsigned char red, unsigned char green, unsigned char blue)
{
	//BGR顺序
	target[2] = red;
	target[1] = green;
	target[0] = blue;
}
//设置需检测的颜色
void ColorDetector::setTargetColor(cv::Vec3b color)
{
	target = color;
}
//获取需检测的颜色
cv::Vec3b ColorDetector::getTargetColor() const
{
	return target;
}
cv::Mat ColorDetector::process(const cv::Mat &image)//核心的处理方法
{
	//按需重新分配二值图像
	//与输入图像的尺寸相同，但是只有一个通道
	result.create(image.rows, image.cols, CV_8U);

	//得到迭代器
	cv::Mat_<cv::Vec3b>::const_iterator it = image.begin<cv::Vec3b>();
	cv::Mat_<cv::Vec3b>::const_iterator itend = image.end<cv::Vec3b>();
	cv::Mat_<uchar>::iterator itout = result.begin<uchar>();
	for (; it != itend; ++it, ++itout)//处理每个像素
	{
		//计算离目标颜色的距离
		if (getDistance(*it) < minDist)
		{
			*itout = 255;
		}
		else
		{
			*itout = 0;
		}
	}
	return result;
}
//int main(int argc, char* argv[])
//{
//	//1.创建图像处理的对象
//	ColorDetector cdetect;
//	//2.读取输入图像
//	cv::Mat image = cv::imread("4.jpg");
//	if (!image.data)
//	{
//		return 0;
//	}
//	imshow("原图", image);
//
//
//	//只显示单一颜色
//	Mat dst;
//	Mat frame, dst;
//	Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
//	Mat kernel_dilite = getStructuringElement(MORPH_RECT, Size(7, 7), Point(-1, -1));
//	//筛选出绿色
//	inRange(image, Scalar(0, 127, 0), Scalar(120, 255, 120), dst);
//	//开操作去噪点
//	morphologyEx(dst, dst, MORPH_OPEN, kernel, Point(-1, -1), 1);
//	//膨胀操作把飞盘具体化的显示出来
//	dilate(dst, dst, kernel_dilite, Point(-1, -1), 2);
//
//	imshow("output video", dst);
//
//
//
//	Mat huidu, erzhi;
//	//灰度处理及二值化
//	cvtColor(image, huidu, CV_BGR2GRAY);
//	namedWindow("gray", 0);
//	cv::imshow("gray", huidu);
//	threshold(huidu, erzhi, 100, 255, THRESH_BINARY);
//	namedWindow("binary", 0);
//	cv::imshow("binary", erzhi) ;
//
//	//保存图片
//	/*string Imag_name1 = "C:\\Users\\善锐刘\\Documents\\Visual Studio 2015\\Projects\\moving_robot\\moving_robot\\gray\\" + to_string(1) + ".jpg";
//	imwrite(Imag_name1, temp);
//	string Imag_name2 = "C:\\Users\\善锐刘\\Documents\\Visual Studio 2015\\Projects\\moving_robot\\moving_robot\\binary\\" + to_string(1) + ".jpg";
//	imwrite(Imag_name2, image);*/
//
//	//3.设置输入参数
//	cdetect.setTargetColor(0, 0 , 0);
//	cv::namedWindow("result",0);
//	//4.处理并显示结果
//	cv::imshow("result", cdetect.process(image));
//
//	cv::waitKey();
//
//	return 0;
//}


//特征提取到matlab

//#include <iostream>  
//#include <opencv2/opencv.hpp>  
//#include <fstream>  
//using namespace std;
//using namespace cv;
//
//int main()
//{
//	Mat src = imread("C:/Users/善锐刘/Pictures/guiji2.png", 0);  //导入图片
//								   //imshow("src", src);  
//
//	Mat dst;
//	threshold(src, dst, 80,255, CV_THRESH_BINARY_INV);  //二值化  
//														  //imshow("dst", dst);  
//
//	int nRows = dst.rows;
//	int nCols = dst.cols;
//
//	ofstream fout("data.txt");
//
//	//按列扫描，求像素和，由于是二值后的图片，没有线条时，该列的像素和为0；扫描到线条时像素大于0  
//	for (int w = 0; w < nCols; w++)
//	{
//		int sum = 0;
//
//		for (int h = 0; h < nRows; h++)
//		{
//			uchar *pRow = dst.ptr<uchar>(h, w); //该列中每个像素的地址  
//			sum += (int)(*pRow);
//
//			if (sum > 0)  //到达了线条的上侧，像素和大于0  
//			{
//				cout << "";   //从上往下找，由于线条很细，目前只判断上边界。
//				cout << w <<" "<< h << endl;
//				fout << w <<" "<< h << endl; //控制台会丢失数据，存到文本不会丢失  
//				sum = 0;
//				break;
//			}
//		}
//	}
//
//	waitKey();
//	cout << endl;
//	system("pause");
//	return 0;
//}


//  颜色处理，提取红色

#include <iostream>
#include<opencv2\opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#define PI 3.1415926
int i = 0;
typedef pair<double, double> pp;
typedef pair<pp, pp> dp;
vector<double> tt;
vector<pp> res;
vector<dp> dres;// 维护一个数组存储一帧图像上面的两组坐标
using namespace cv;
using namespace std;
double to_rad(CvPoint p1, CvPoint p2) {
	float t = (p1.x*p2.x + p1.y + p2.y) / (sqrt(pow(p1.x,2)+pow(p1.y,2))*sqrt(pow(p2.x,2)+pow(p2.y,2)));
	return acos(t);
}
void RGB2HSV(double red, double green, double blue, double& hue, double& saturation, double& intensity)
{
	double r, g, b;
	double h, s, i;

	double sum;
	double minRGB, maxRGB;
	double theta;

	r = red / 255.0;
	g = green / 255.0;
	b = blue / 255.0;

	minRGB = ((r<g) ? (r) : (g));
	minRGB = (minRGB<b) ? (minRGB) : (b);

	maxRGB = ((r>g) ? (r) : (g));
	maxRGB = (maxRGB>b) ? (maxRGB) : (b);

	sum = r + g + b;
	i = sum / 3.0;

	if (i<0.001 || maxRGB - minRGB<0.001)
	{
		h = 0.0;
		s = 0.0;
	}
	else
	{
		s = 1.0 - 3.0*minRGB / sum;
		theta = sqrt((r - g)*(r - g) + (r - b)*(g - b));
		theta = acos((r - g + r - b)*0.5 / theta);
		if (b <= g)
			h = theta;
		else
			h = 2 * PI - theta;
		if (s <= 0.01)
			h = 0;
	}
	hue = (int)(h * 180 / PI);
	saturation = (int)(s * 100);
	intensity = (int)(i * 100);
}
Mat picture_red(Mat input)
{
	Mat frame;
	Mat srcImg = input;
	frame = srcImg;
	waitKey(1);
	int width = srcImg.cols;
	int height = srcImg.rows;

	int x, y;
	double B = 0.0, G = 0.0, R = 0.0, H = 0.0, S = 0.0, V = 0.0;
	Mat vec_rgb = Mat::zeros(srcImg.size(), CV_8UC1);
	// 遍历所有像素点
	for (x = 0; x < height; x++)
	{
		for (y = 0; y < width; y++)
		{
			B = srcImg.at<Vec3b>(x, y)[0];
			G = srcImg.at<Vec3b>(x, y)[1];
			R = srcImg.at<Vec3b>(x, y)[2];
			RGB2HSV(R, G, B, H, S, V);
			//红色范围，范围参考的网上。可以自己调
			//if ((H >= 312 && H <= 360) && (S >= 17 && S <= 100) && (V>18 && V < 100))//粉红色
			//if (((H >= 0 && H <= 10)|| (H >= 156 && H <= 180)) && (S >= 43 && S <= 255) && (V>0 && V < 46))
			if (((H >= 0 && H <= 10) || (H >= 156 && H <= 180)) && (S >= 43 && S <= 255) && (V>20 && V < 46))//红色
				vec_rgb.at<uchar>(x, y) = 255;
			/*cout << H << "," << S << "," << V << endl;*/
		}
	}
	/*imshow("hsv", vec_rgb);*/  
	string Imag_name2 = "C:\\Users\\善锐刘\\Documents\\Visual Studio 2015\\Projects\\moving_robot\\moving_robot\\gray\\" + to_string(1) + ".jpg";
	imwrite(Imag_name2, vec_rgb);
	return vec_rgb;
}
CvPoint O_x1y1(Mat in, double *x1, double *y1, double *x2, double *y2)
{
	i++;
	Mat matSrc = in;
	/*Mat matSrc = imread("qwer9.png", 0);*/

	string Imag_name4 = "C:\\Users\\善锐刘\\Documents\\Visual Studio 2015\\Projects\\moving_robot\\moving_robot\\binary\\" + to_string(4) + ".jpg";
	imwrite(Imag_name4, matSrc);

	//GaussianBlur(matSrc, matSrc, Size(10, 10), 0);//高斯滤波，除噪点

	/*namedWindow("高斯滤波", 0);
	imshow("高斯滤波", matSrc);*/
	vector<vector<Point> > contours;//contours的类型，双重的vector
	vector<Vec4i> hierarchy;//Vec4i是指每一个vector元素中有四个int型数据。
							//阈值
	//灰度处理
	/*Mat erzhi;
	cvtColor(matSrc, hui, CV_BGR2GRAY);
	namedWindow("灰度图", 0);
	cv::imshow("灰度图", hui);*/

	Mat erzhi;
	threshold(matSrc, erzhi, 0, 255, THRESH_BINARY);//图像二值化
	string Imag_name1 = "C:\\Users\\善锐刘\\Documents\\Visual Studio 2015\\Projects\\moving_robot\\moving_robot\\binary\\" + to_string(1) + ".jpg";
	imwrite(Imag_name1, erzhi);

	/*namedWindow("二值化", 0);
	imshow("二值化", erzhi);*/
	//寻找轮廓，这里注意，findContours的输入参数要求是二值图像，二值图像的来源大致有两种，第一种用threshold，第二种用canny
	
	// 形态学处理
	Mat op;
	Mat elem = getStructuringElement(MORPH_RECT,Size(15,15));
	erode(erzhi, op,elem);
	string Imag_name3 = "C:\\Users\\善锐刘\\Documents\\Visual Studio 2015\\Projects\\moving_robot\\moving_robot\\binary\\" + to_string(2) + ".jpg";
	imwrite(Imag_name3, op);

	findContours(erzhi.clone(), contours, hierarchy, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));
	
	/// 计算矩
	vector<Moments> mu(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		mu[i] = moments(contours[i], false);
	}

	///  计算矩中心:
	vector<Point2f> mc(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		mc[i] = Point2f(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
	}


	///// 绘制轮廓
	//Mat drawing = Mat::zeros(matSrc.size(), CV_8UC1); //后面声明为全局变量
	//for (int i = 0; i < contours.size(); i++)
	//{
	//	Scalar color = Scalar(255);
	//	//drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());//绘制轮廓函数
	//	circle(drawing, mc[i], 4, color, -1, 8, 0);
	//}
	//namedWindow("outImage", 0);
	//imshow("outImage", drawing);


	//保证小点在上
	if (mc[0].y < mc[contours.size() - 1].y)
	{
		*x1 = mc[0].x;
		*y1 = mc[0].y;
		*x2 = mc[contours.size() - 1].x;
		*y2 = mc[contours.size() - 1].y;
	}
	else {
		*x2 = mc[0].x;
		*y2 = mc[0].y;
		*x1 = mc[contours.size() - 1].x;
		*y1 = mc[contours.size() - 1].y;
	}
	

	//计算向量
	CvPoint center;
	center.x = *x1 - *x2;
	center.y = *y1 - *y2;
	// 打开模式可省略 
	return center;
}
int main()
{
	//VideoCapture video("12.mp4 ");
	//if (!video.isOpened())  //对video进行异常检测  
	//{
	//	cout << "video open error!" << endl;
	//	return 0;
	//}
	//Mat src;
	//
	//int frameCount = video.get(CAP_PROP_FRAME_COUNT);//获取帧数
	////cout << frameCount << endl;
	//double FPS = video.get(CAP_PROP_FPS);	//获取FPS  
	//Mat frame;								//存储帧  
	//Mat temp;								//存储前一帧图像  
	//Mat result;								//存储结果图像  
	//for (int i = 0; i < frameCount; i++)
	//{	
	//	video >> frame;						//读帧进frame  
	//	//imshow("frame", frame);
	//	if (frame.empty())					//对帧进行异常检测  
	//	{
	//		cout << "frame is empty!" << endl;
	//		break;
	//	}			
	//	 Mat matSrc = frame.clone();
	//	// 一张图片所做的事情
	//	double xx1, yy1, xx2, yy2;
	//	//double x1, y1, x2, y2;
	//	Mat middle = picture_red(matSrc);
	//	CvPoint pt = O_x1y1(middle, &xx1, &yy1, &xx2, &yy2);
	//	//保存到CSV
	//	//保存数据到CSV 
	//	/*cout<< pt.x << ',' << pt.y << endl;
	//	namedWindow("原图", 0);
	//	imshow("原图", matSrc);
	//	namedWindow("red", 0);
	//	imshow("red", picture_red(matSrc));*/
	//	
	//	// 滤波程序
	//	//if ((abs(xx1 - xx2) > 200) || abs(yy1-yy2)>200) continue;
	//	if ((pow((xx1-xx2),2)+pow((yy1-yy2),2))>pow(200,2)) continue;

	//	//时间戳
	//	double ttt = video.get(CAP_PROP_POS_MSEC)/1000;
	//	tt.emplace_back(ttt);
		
	//	res.emplace_back(make_pair(pt.x, pt.y));
	//	//存储坐标点
	//	dres.emplace_back(make_pair(xx1, yy1), make_pair(xx2, yy2));
	//	
	//	 
	//}
	//////保存数据到CSV 
	//ofstream outFile;
	//outFile.open("data1.csv", ios::out | ios::trunc);
	//for (auto x : res) {
	//	outFile << x.first << "," << x.second << endl;
	//}
	//outFile.close();

	////保存所有数据;
	//outFile.open("data_all.csv", ios::out | ios::trunc);
	//for (auto x : dres) {
	//	outFile << x.first.first << "," << x.first.second<<","<<x.second.first<<","<<x.second.second << endl;
	//}
	//outFile.close();

	////保存一个点数据
	//outFile.open("data_one.csv", ios::out | ios::trunc);
	//for (auto x : dres) {
	//	outFile << x.first.first << "," << x.first.second << endl;
	//}
	//outFile.close();

	////保存事件戳
	//outFile.open("data_time.csv", ios::out | ios::trunc);
	//for (auto x : tt) {
	//	outFile << x << endl;
	//}
	//outFile.close();

	//cout << "总共调用" << i << "张照片" << endl;
	//waitKey();
	//return 0;


	// 一张图片所做的事情
	double xx1, yy1, xx2, yy2;
	double x1, y1, x2, y2;
	Mat matSrc = imread("13.jpg");

	Mat matSrc1;
	GaussianBlur(matSrc, matSrc1, Size(3, 3), 0);//高斯滤波，除噪点
	string Imag_name3 = "C:\\Users\\善锐刘\\Documents\\Visual Studio 2015\\Projects\\moving_robot\\moving_robot\\binary\\" + to_string(3) + ".jpg";
	imwrite(Imag_name3, matSrc1);

	Mat middle = picture_red(matSrc1);
	CvPoint pt=O_x1y1(middle, &xx1, &yy1, &xx2, &yy2);
	//保存到CSV
	//保存数据到CSV 
	/*cout<< pt.x << ',' << pt.y << endl;
	namedWindow("原图", 0);
	imshow("原图", matSrc);
	namedWindow("red", 0);
	imshow("red", picture_red(matSrc));*/

	////保存数据到CSV 
	ofstream outFile;
	outFile.open("data1.csv", ios::out|ios::trunc);            // 打开模式可省略 
	outFile << pt.x << "," << pt.y << endl;
	outFile.close();

	waitKey(0);
	return 0;
}





//  单张图片提取背景
///运动物体检测——帧差法  

////#include"opencv2/opencv.hpp"  
////#include<opencv2/highgui/highgui.hpp>
////#include<opencv2/core/core.hpp>
////#include<opencv2/imgproc/imgproc.hpp>
////
////using namespace cv;
////#include <iostream>  
////using namespace std;
//////运动物体检测函数声明  
////Mat MoveDetect(Mat temp, Mat frame);
////int main()
////{
////	 //VideoCapture video(0);//定义VideoCapture类video  
////	VideoCapture video("liu.mp4 ");
////	 if (!video.isOpened())  //对video进行异常检测  
////	 {  
////	    cout << "video open error!" << endl;  
////	    return 0;  
////	 }  
////	while (1)
////	{
////		int frameCount = video.get(CAP_PROP_FRAME_COUNT);//获取帧数
////		cout << frameCount << endl;
////		double FPS = video.get(CAP_PROP_FPS);//获取FPS  
////		Mat frame;//存储帧  
////		Mat temp;//存储前一帧图像  
////		Mat result;//存储结果图像  
////		for (int i = 0; i < frameCount; i++)
////		{
////
////			video >> frame;//读帧进frame  
////			imshow("frame", frame);
////			if (frame.empty())//对帧进行异常检测  
////			{
////				cout << "frame is empty!" << endl;
////				break;
////			}
////			if (i == 0)//如果为第一帧（temp还为空）  
////			{
////				result = MoveDetect(frame, frame);//调用MoveDetect()进行运动物体检测，返回值存入result  
////				imshow("一帧",result);
////			}
////			else//若不是第一帧（temp有值了）  
////			{
////				result = MoveDetect(temp, frame);//调用MoveDetect()进行运动物体检测，返回值存入result  
////				
////			}
////			imshow("result", result);
////			if (waitKey(1000.0 / FPS) == 27)//按原FPS显示  
////			{
////				cout << "ESC退出!" << endl;
////				break;
////			}
////			temp = frame.clone();
////		}
////	}
////	return 0;
////
////}
////Mat MoveDetect(Mat temp, Mat frame)
////{
////	Mat result = frame.clone();
////	//1.将background和frame转为灰度图  
////	Mat gray1, gray2;
////	cvtColor(temp, gray1, CV_BGR2GRAY);
////	cvtColor(frame, gray2, CV_BGR2GRAY);
////	//2.将background和frame做差  
////	Mat diff;
////	absdiff(gray1, gray2, diff);
////	imshow("diff", diff);
////	//3.对差值图diff_thresh进行阈值化处理  
////	Mat diff_thresh;
////	threshold(diff, diff_thresh, 50, 255, CV_THRESH_BINARY);
////	imshow("diff_thresh", diff_thresh);
////	//4.腐蚀  
////	Mat kernel_erode = getStructuringElement(MORPH_RECT, Size(3, 3));
////	Mat kernel_dilate = getStructuringElement(MORPH_RECT, Size(18, 18));
////	erode(diff_thresh, diff_thresh, kernel_erode);
////	imshow("erode", diff_thresh);
////	//5.膨胀  
////	dilate(diff_thresh, diff_thresh, kernel_dilate);
////	imshow("dilate", diff_thresh);
////	//6.查找轮廓并绘制轮廓  
////	vector<vector<Point> > contours;
////	findContours(diff_thresh, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
////	drawContours(result, contours, -1, Scalar(0, 0, 255), 2);//在result上绘制轮廓  
////															 //7.查找正外接矩形  
////	vector<Rect> boundRect(contours.size());
////	for (int i = 0; i < contours.size(); i++)
////	{
////		boundRect[i] = boundingRect(contours[i]);
////		rectangle(result, boundRect[i], Scalar(0, 255, 0), 2);//在result上绘制正外接矩形  
////	}
////	return result;//返回result  
////}
//
//
////  提取图形的形心
//#include<iostream>
//#include<fstream>
//#include<string>
//using namespace std;
////CvPoint GetCenterPoint(Mat &src)
////{
////	//CvArr* s = (CvArr*)&src;
////	int i, j;
////	int x0 = 0, y0 = 0, sum = 0;
////	CvPoint center;
////	int  pixel;
////	cout << endl << src.rows << " " << src.cols << endl;
////	for (i = 0; i<src.rows; i++)
////		for (j = 0; j<src.cols; j++)
////		{
////			//pixel = cvGet2D(s, j, i);
////			pixel = src.at<uchar>(i,j);
////			if (pixel == 1)
////			{
////				x0 = x0 + i;
////				y0 = y0 + j;
////				sum = sum + 1;
////			}
////		}
////	center.x = x0 / sum;
////	center.y = y0 / sum;
////	return center;
////}
////void main()
////{
////	Mat src = imread("src.jpg");
////	CvPoint CenterPoint;
////	CenterPoint = GetCenterPoint(src);
////	cout << "CenterPoint.x: " << CenterPoint.x << "\t" << "CenterPoint.y:" << CenterPoint.y << endl;
////	circle(src, CenterPoint, 3, Scalar(0, 255, 0), 1, 8, 3);
////	namedWindow("Center", WINDOW_AUTOSIZE);
////	imshow("Center", src);
////	waitKey();
////}
//
//
//CvPoint GetCenterPoint(Mat &src)
//{
//	//CvArr* s = (CvArr*)&src;
//	int i, j;
//	int x0 = 0, y0 = 0, sum = 0;
//	CvPoint center;
//	int  pixel;
//	cout << endl << src.rows << " " << src.cols << endl;
//	for (i = 0; i<src.rows; i++)
//		for (j = 0; j<src.cols; j++)
//		{
//			//pixel = cvGet2D(s, j, i);
//			pixel = src.at<uchar>(i, j);
//			//cout << pixel << endl;
//			if (pixel == 0)
//			{
//				x0 = x0 + i;
//				y0 = y0 + j;
//				sum = sum + 1;
//			}
//		}
//	center.x = x0 / sum;
//	center.y = y0 / sum;
//	return center;
//}
//void findHome(Mat &src) {
//	int div = 64;
//	for(int i=0;i<src.rows;++i)
//		for (int j = 0; j < src.cols; ++j) {
//			//在这里访问每个通道的元素,注意，成员函数at(int y,int x)的参数
//			src.at<Vec3b>(i, j)[0] = src.at<Vec3b>(i, j)[0] / div*div + div / 2;
//			src.at<Vec3b>(i, j)[1] = src.at<Vec3b>(i, j)[1] / div*div + div / 2;
//			src.at<Vec3b>(i, j)[2] = src.at<Vec3b>(i, j)[2] / div*div + div / 2;
//			if (src.at<Vec3b>(i, j)[0] == 0 && src.at<Vec3b>(i, j)[1] == 0 && src.at<Vec3b>(i, j)[2] == 255) {
//				src.at<Vec3b>(i, j)[0] = 1;
//				src.at<Vec3b>(i, j)[1] = 1;
//				src.at<Vec3b>(i, j)[2] = 1;
//			}
//			else {
//				src.at<Vec3b>(i, j)[0] = 0;
//				src.at<Vec3b>(i, j)[1] = 0;
//				src.at<Vec3b>(i, j)[2] = 0;
//			}
//
//		}
//}
////int main() 
////{
////	//视频处理
////
////	//VideoCapture video("liu.mp4 ");
////	//if (!video.isOpened())  //对video进行异常检测  
////	//{
////	//	cout << "video open error!" << endl;
////	//	return 0;
////	//}
////	//Mat src;
////	//while (1)
////	//{
////	//	int frameCount = video.get(CAP_PROP_FRAME_COUNT);//获取帧数
////	//	cout << frameCount << endl;
////	//	double FPS = video.get(CAP_PROP_FPS);	//获取FPS  
////	//	Mat frame;								//存储帧  
////	//	Mat temp;								//存储前一帧图像  
////	//	Mat result;								//存储结果图像  
////	//	for (int i = 0; i < frameCount; i++)
////	//	{	
////	//		video >> frame;						//读帧进frame  
////	//		imshow("frame", frame);
////	//		if (frame.empty())					//对帧进行异常检测  
////	//		{
////	//			cout << "frame is empty!" << endl;
////	//			break;
////	//		}
////	//		
////	//		temp = frame.clone();
////	//		
////	//		
////	//	}
////	//}
////	//return 0;
////
////
////	//图片处理
////
////	//读取输入图像
////	cv::Mat image = cv::imread("9.jpg");
////	Mat temp;
////	if (!image.data)
////	{
////		return 0;
////	}
////
////	//灰度处理及二值化
////	cvtColor(image, temp, CV_BGR2GRAY);
////	cv::imshow("gray", temp);
////	threshold(temp, image, 100, 255, THRESH_BINARY_INV);
////	cv::imshow("binary", image);
////
////	//膨胀操作
////	/*Mat pz;
////	Mat element = getStructuringElement(MORPH_RECT, Size(50, 50));
////	dilate(image, pz, element);
////	cv::imshow("膨胀",pz);*/
////
////	//保存图片
////	string Imag_name1 = "C:\\Users\\善锐刘\\Documents\\Visual Studio 2015\\Projects\\moving_robot\\moving_robot\\gray\\" + to_string(1) + ".jpg";
////	imwrite(Imag_name1, temp);
////	string Imag_name2 = "C:\\Users\\善锐刘\\Documents\\Visual Studio 2015\\Projects\\moving_robot\\moving_robot\\binary\\" + to_string(1) + ".jpg";
////	imwrite(Imag_name2, image);
////
////	// 寻找中点
////	CvPoint CenterPoint;
////	CenterPoint = GetCenterPoint(image);
////	cout << "CenterPoint.x: " << CenterPoint.x << "\t" << "CenterPoint.y:" << CenterPoint.y << endl;
////	/*YKcircle(image, CvPoint(419, 268), 40, 1);
////	circle(image, CvPoint(419,268), 80, Scalar(255, 255, 255), 1, 8, 3);
////	circle(image, CvPoint(0, 0), 80, Scalar(255, 255, 255), 1, 8, 3);
////	circle(image, CvPoint(805, 536), 80, Scalar(255, 255, 255), 1, 8, 3);
////	namedWindow("Center", WINDOW_AUTOSIZE);
////	imshow("Center", image);*/
////
////	// 霍夫圆变换求圆心坐标
////	
////	//进行霍夫圆变换1
////	Mat bf;//对灰度图像进行双边滤波
////	const int kvalue = 15;
////	bilateralFilter(temp, bf, kvalue, kvalue * 2, kvalue / 2);
////	imshow("灰度双边滤波处理", bf);
////	//声明一个三通道图像，像素值全为0，用来将霍夫变换检测出的圆画在上面
////	Mat dst(image.size(), image.type());
////	dst = Scalar::all(0);
////
////	vector<Vec3f> circles;//声明一个向量，保存检测出的圆的圆心坐标和半径,
////	HoughCircles(bf, circles, CV_HOUGH_GRADIENT, 2, bf.rows / 4, 130, 38, 10, 50);//霍夫变换检测圆//霍夫变换检测圆
////
////	cout << "x=\ty=\tr=" << endl;
////	for (size_t i = 0; i < circles.size(); i++)//把霍夫变换检测出的圆画出来
////	{
////		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
////		int radius = cvRound(circles[i][2]);
////
////		circle(image, center, 0, Scalar(0, 255, 0), -1, 8, 0);
////		circle(image, center, radius, Scalar(0, 0, 255), 1, 8, 0);
////
////		cout << cvRound(circles[i][0]) << "\t" << cvRound(circles[i][1]) << "\t"
////			<< cvRound(circles[i][2]) << endl;//在控制台输出圆心坐标和半径				
////	}
////
////	imshow("特征提取", image);
////
////	
////
////	//保存数据到CSV 
////	ofstream outFile;
////	outFile.open("data.csv", ios::out);            // 打开模式可省略 
////	outFile << CenterPoint.x << ',' << CenterPoint.y << endl;
////	outFile.close();
////
////	cv::waitKey();
////
////	return 0;
////}



