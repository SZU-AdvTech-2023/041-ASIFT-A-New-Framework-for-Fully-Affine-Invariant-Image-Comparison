#include <stdio.h>
#include <stdlib.h>
#include <io.h>
#include <fstream>
#include "ASIFT_CUDA.cuh"
#include <opencv2/core/ocl.hpp>
#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <time.h>
using namespace cv;
using namespace std;

const bool file_out = false;

const int NUM_OF_TILT = 7;

const vector<string> folders = { "index0_idx0","index0_idx1","index0_idx2","index0_idx3","index0_idx4",
								"index0_idx5","index0_idx6","index0_idx7","index0_idx8",
								"index1_idx0","index1_idx1","index1_idx2","index1_idx3","index1_idx4",
								"index1_idx5","index1_idx6","index1_idx7","index1_idx8",
								"index3_idx0","index3_idx1","index3_idx2","index3_idx3","index3_idx4",
								"index3_idx5","index3_idx6","index3_idx7","index3_idx8","index3_idx9",
								"index3_idx10","index3_idx11",
								"index7_idx0","index7_idx1","index7_idx2","index7_idx3","index7_idx4",
								"index7_idx5","index7_idx6","index7_idx7","index7_idx8","index7_idx9",
								"index7_idx10","index7_idx11","index7_idx12","index7_idx13","index7_idx14","index7_idx15" };

const vector<string> files = { "0000","0001","0002" };

void Asift_test(string folder_name)
{
	string index1 = ".\\Open\\test\\" + folder_name + "\\Image\\" + files[0] + ".tif";
	string index2 = ".\\Open\\test\\" + folder_name + "\\Image\\" + files[1] + ".tif";
	string index3 = ".\\Open\\test\\" + folder_name + "\\Image\\" + files[2] + ".tif";

	string res_file = ".\\RESULT\\";
	
	Mat img1, img2;

	for (int i = 0; i < 3; i++)
	{
		string res_file_name;
		FILE* fp;
		if (i == 0)
		{
			res_file_name = res_file + folder_name + "_" + files[0] + "_" + files[1] + ".tif";
			string file_out = res_file + folder_name + "_" + "Out0000-0001.txt";
			fp = freopen((char*)file_out.c_str(), "w", stdout);

			img1 = imread(index1, 2);
			img2 = imread(index2, 2);
		}
		else if (i == 1)
		{
			res_file_name = res_file + folder_name + "_" + files[0] + "_" + files[2] + ".tif";
			string file_out = res_file + folder_name + "_" + "Out0000-0002.txt";
			fp = freopen((char*)file_out.c_str(), "w", stdout);

			img1 = imread(index1, 2);
			img2 = imread(index3, 2);
		}
		else if (i == 2)
		{
			res_file_name = res_file + folder_name + "_" + files[1] + "_" + files[2] + ".tif";
			string file_out = res_file + folder_name + "_" + "Out0001-0002.txt";
			fp = freopen((char*)file_out.c_str(), "w", stdout);

			img1 = imread(index2, 2);
			img2 = imread(index3, 2);
		}

		std::vector<cv::KeyPoint> kpts1, kpts2;
		cv::Mat des1, des2;

		asift::DetectAndComputeAsift(img1, kpts1, des1, NUM_OF_TILT);
		//asift::SplitTest(img1, kpts1, des1);
		//asift::AdaptiveSegmentation(img1, kpts1, des1, NUM_OF_TILT);
		//std::cout << "Size of des1: [" << des1.cols << "*" << des1.rows << "]\n";
		//printf("----------------------------------------------------------------\n\n");

		asift::DetectAndComputeAsift(img2, kpts2, des2, NUM_OF_TILT);
		//asift::SplitTest(img2, kpts2, des2);
		//asift::AdaptiveSegmentation(img2, kpts2, des2, NUM_OF_TILT);
		//std::cout << "Size of des2: [" << des2.cols << "*" << des2.rows << "]\n";
		//printf("----------------------------------------------------------------\n\n");

		//cv::FlannBasedMatcher fbmlatcher;
		cv::Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher();
		cv::cuda::GpuMat des1_gpu(des1), des2_gpu(des2);
		std::vector< std::vector< cv::DMatch > > matches;

		clock_t start=clock(), end;
		//fbmlatcher.knnMatch(des1, des2, matches, 2);
		matcher->knnMatch(des1_gpu,des2_gpu,matches,2);
		end = clock();
		double duration = (double)(end - start) / CLOCKS_PER_SEC;
		printf("match: %.2lf\n", duration);

		std::vector<cv::DMatch> good_matches, ransacti;
		for (size_t j = 0; j < matches.size(); ++j) {
			if (matches[j].size() >= 2) {
				if (matches[j][0].distance < 0.8 * matches[j][1].distance) {
					good_matches.push_back(matches[j][0]);
				}
			}
		}

		if (good_matches.size() == 0) return;

		vector<Point2f> kp1, kp2;
		for (int j = 0; j < good_matches.size(); j++)
		{
			kp1.push_back(kpts1[good_matches[j].queryIdx].pt);
			kp2.push_back(kpts2[good_matches[j].trainIdx].pt);
		}

		vector<uchar> RansacStatus;
		Mat m_Homography1 = findHomography(kp2, kp1, RansacStatus, USAC_MAGSAC, 3);
		for (size_t i = 0; i < RansacStatus.size(); i++)
		{
			if (RansacStatus[i] != 0)
			{
				ransacti.push_back(good_matches[i]);
			}
		}
		cout << "Matched: " << good_matches.size() << endl;
		cout << "Filtered: " << ransacti.size() << endl;
		Mat res;
		img1.convertTo(img1, CV_8UC1);
		img2.convertTo(img2, CV_8UC1);
		drawMatches(img1, kpts1, img2, kpts2, ransacti, res);
		imwrite(res_file_name, res);

		fclose(fp);
	}
}
void help()
{
	cout << "ASIFT_CUDA [detectOption] [image1] [image2] [resultPath]                    " << endl;
	cout << "    [dectectOption]      -origin                 original asift             " << endl;
	cout << "                         -openmp                 original asift with openmp " << endl;
	cout << "                         -cudasift               using cudasift to replace  " << endl;
	cout << "                                                 sift in original asift     " << endl;
	cout << "                         -openmp_cudasift        using cudasift and enable  " << endl;
	cout << "                                                 openmp(image maybe splited " << endl;
	cout << "                                                 due to insufficient of gpu " << endl;
	cout << "                                                 memory)                    " << endl;
	cout << "                         -rtg                    running image rotate, tilt " << endl;
	cout << "                                                 and gaussian blur on gpu   " << endl;
	cout << "                         -rtg_cudasift           running image processing on" << endl;
	cout << "                                                 gpu and using cudasift     " << endl;
	cout << "                         -openmp_rtg_cudasift    running image processing on" << endl;
	cout << "                                                 gpu, using cudasift and    " << endl;
	cout << "                                                 enable openmp(image may be " << endl;
	cout << "                                                 splited due to insufficient" << endl;
	cout << "                                                 of gpu memory)             " << endl;
	cout << "                         -npp_cudasift           running image processing on" << endl;
	cout << "                                                 gpu using Nvidia 2D Image  " << endl;
	cout << "                                                 And Signal Performance     " << endl;
	cout << "                                                 Primitives and using       " << endl;
	cout << "                                                 cudasift                   " << endl;
}

int main(int argc, char* argv[])
{
	if (argc != 5)
	{
		help();
		return 0;
	}
	string detectOption = argv[1];
	string imgPath1 = argv[2];
	string imgPath2 = argv[3];
	string outputPath = argv[4];

	int opt = 0;
	if(detectOption=="-origin")						opt = 1;
	else if(detectOption=="-openmp")				opt = 2;
	else if(detectOption=="-cudasift")				opt = 3;
	else if(detectOption=="-openmp_cudasift")		opt = 4;
	else if(detectOption=="-rtg_cudasift")			opt = 5;
	else if(detectOption=="-openmp_rtg_cudasift")	opt = 6;
	else if(detectOption=="-npp_cudasift")			opt = 7;
	else
	{
		cout << "Wrong Detect Option!" << endl;
		return 0;
	}
	string timeFile = outputPath + "time.txt";
	string imageFile = outputPath + "result.tif";
	FILE* file = freopen(timeFile.c_str(), "w", stdout);

	Mat img1 = imread(imgPath1, 2);
	Mat img2 = imread(imgPath2, 2);
	vector<KeyPoint> kpts1, kpts2;
	Mat des1, des2;

	asift::UltimateTest(img1, kpts1, des1, opt, NUM_OF_TILT);

	asift::UltimateTest(img2, kpts2, des2, opt, NUM_OF_TILT);

	cv::Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher();
	cv::cuda::GpuMat des1_gpu(des1), des2_gpu(des2);
	std::vector< std::vector< cv::DMatch > > matches;

	chrono::time_point<std::chrono::high_resolution_clock> start, end;
	start = chrono::high_resolution_clock::now();

	matcher->knnMatch(des1_gpu, des2_gpu, matches, 2);

	end = chrono::high_resolution_clock::now();

	double duration = chrono::duration_cast<chrono::microseconds>(end - start).count() * 0.000001;
	cout << "MATCH: " << duration << " s" << endl;

	std::vector<cv::DMatch> good_matches, ransacti;
	for (size_t j = 0; j < matches.size(); ++j) {
		if (matches[j].size() >= 2) {
			if (matches[j][0].distance < 0.8 * matches[j][1].distance) {
				good_matches.push_back(matches[j][0]);
			}
		}
	}
	if (good_matches.size() == 0) return;

	vector<Point2f> kp1, kp2;
	for (int j = 0; j < good_matches.size(); j++)
	{
		kp1.push_back(kpts1[good_matches[j].queryIdx].pt);
		kp2.push_back(kpts2[good_matches[j].trainIdx].pt);
	}
	vector<uchar> RansacStatus;
	Mat m_Homography1 = findHomography(kp2, kp1, RansacStatus, USAC_MAGSAC, 3);
	for (size_t i = 0; i < RansacStatus.size(); i++)
	{
		if (RansacStatus[i] != 0)
		{
			ransacti.push_back(good_matches[i]);
		}
	}
	cout << "Matched: " << good_matches.size() << endl;
	cout << "Filtered: " << ransacti.size() << endl;

	Mat res;
	img1.convertTo(img1, CV_8UC1);
	img2.convertTo(img2, CV_8UC1);
	drawMatches(img1, kpts1, img2, kpts2, ransacti, res);
	imwrite(imageFile, res);
	fclose(file);
	/*
	TODO: 1. make a -rtg only function
		  2. image split using npp
	*/

	/*InitCuda(0);
	ocl::setUseOpenCL(false);
	
	for (string i : folders)
	{
		Asift_test(i);
	}*/
	return 0;
}
