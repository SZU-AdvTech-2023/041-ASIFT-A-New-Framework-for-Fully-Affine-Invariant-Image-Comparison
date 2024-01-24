/**
 * @file asift.h
 * @author Gareth Wang (gareth.wang@hotmail.com)
 * @brief Interface of ASIFT feature extractor unified into OpenCV modules.
 * @version 0.1
 * @date 2019-10-30
 *
 * @copyright Copyright (c) 2019
 *
 */
#ifndef _ASIFT_H_
#define _ASIFT_H_
#include <chrono>
#include <opencv2/opencv.hpp>
#include "compute_asift_keypoints.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"



void getFiles(string foler, vector<string>& files);
namespace asift {

	std::vector<float> ConvertImageFromMatToVector(const cv::Mat& image) {

		int width = image.cols;
		int height = image.rows;

		float* pdata;
		cv::Mat gray_image;
		if (image.channels() == 1) {
			image.copyTo(gray_image);
			gray_image.convertTo(gray_image, CV_32F);
		}
		else {
			std::vector<cv::Mat> bgr;
			cv::split(image, bgr);
			bgr[0].convertTo(bgr[0], CV_32F);
			bgr[1].convertTo(bgr[1], CV_32F);
			bgr[2].convertTo(bgr[2], CV_32F);

			gray_image = (6969 * bgr[2] + 23434 * bgr[1] + 2365 * bgr[0]) / 32768;
		}
		pdata = (float*)gray_image.data;
		std::vector<float> vec_image(pdata, pdata + width * height);

		return vec_image;
	}

	void DetectAndComputeAsift(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors, int tilt, int dopt = 0) {
		std::vector<float> ipixels = ConvertImageFromMatToVector(image);
		int w, h;
		w = image.cols;
		h = image.rows;

		float zoom = 0;
		int wS1 = 0, hS1 = 0;
		vector<float> ipixels_zoom;
		

		ipixels_zoom.resize(w * h);
		ipixels_zoom = ipixels;
		wS1 = w;
		hS1 = h;
		zoom = 1;

		///// Compute ASIFT keypoints
		// number N of tilts to simulate t = 1, \sqrt{2}, (\sqrt{2})^2, ..., {\sqrt{2}}^(N-1)
		int num_of_tilts = tilt;

		int verb = 0;

		vector< vector< keypointslist > > keys;

		int num_keys = 0;

		std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
		start = std::chrono::high_resolution_clock::now();

		if (dopt == 1)
			num_keys = compute_asift_keypoints(ipixels_zoom, wS1, hS1, num_of_tilts, verb, keys);
		else if (dopt == 2)
			num_keys = compute_asift_keypoints_openmp(ipixels_zoom, wS1, hS1, num_of_tilts, verb, keys);
		else if (dopt == 3)
			num_keys = compute_asift_keypoints_CudaSift(ipixels_zoom, wS1, hS1, num_of_tilts, verb, keys);
		else if (dopt == 4)
			num_keys = compute_asift_keypoints_openmp_CudaSift(ipixels_zoom, wS1, hS1, num_of_tilts, verb, keys);
		else if (dopt == 5)
			num_keys = compute_asift_keypoints_rtg_CudaSift(ipixels_zoom, wS1, hS1, num_of_tilts, verb, keys);
		else if (dopt == 6)
			num_keys = compute_asift_keypoints_rtg_openmp_CudaSift(ipixels_zoom, wS1, hS1, num_of_tilts, verb, keys);
		else if (dopt == 7)
			num_keys = compute_asift_keypoints_npp_CudaSift(ipixels_zoom, wS1, hS1, num_of_tilts, verb, keys, image.step);

		end = std::chrono::high_resolution_clock::now();

		double duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 0.000001;
		if (dopt != 6)
			cout << "ASIFT: " << duration << " s" << endl;

		keypoints.clear();
		descriptors.release();
		for (int tt = 0; tt < (int)keys.size(); tt++) {
			for (int rr = 0; rr < (int)keys[tt].size(); rr++)
			{
				keypointslist::iterator ptr = keys[tt][rr].begin();
				for (int i = 0; i < (int)keys[tt][rr].size(); i++, ptr++)
				{
					cv::KeyPoint keypt(zoom * ptr->x, zoom * ptr->y, -1, ptr->angle);
					keypoints.push_back(keypt); // Convert keypoints from user-define to OpenCV

					cv::Mat des = cv::Mat(1, (int)VecLength, CV_32F, ptr->vec).clone();
					// Normalize the descriptor
					des = des / cv::norm(des);
					descriptors.push_back(des); // Save descriptors of the keypoints
				}
			}
		}
	}

	long long get_total_turn(int tilt)
	{
		long long counter_sim = 0;
		for (int tt = 1; tt <= tilt; tt++)
		{
			float t = 1.f * pow(sqrt(2.f), tt - 1);

			if (t == 1) counter_sim++;
			else
			{
				int num_rot1 = round(10 * t / 2);
				if (num_rot1 % 2 == 1) num_rot1++;

				num_rot1 /= 2;
				counter_sim += num_rot1;
			}
		}
		return counter_sim;
	}

	long long get_memory_needed(int width, int height, int tilt)
	{
		long long total = 0;
		long long image_size = width * height * sizeof(float);
		total += (image_size * (get_total_turn(tilt) * 5));//临时图像三张+CudaSift所需两倍临时内存
		//total += width * 10000 * sizeof(float);//高斯模糊所需内存
		//total += image_size * (get_total_turn(tilt) * 5 - 5);//倾斜操作所需部分内存

		return total + image_size;
	}

	void AdaptiveSegmentation(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors, int tilt, int dopt = 0)
	{
		int num_of_tilt = tilt;//旋转模拟次数

		size_t avail, total;
		cudaMemGetInfo(&avail, &total);//获取当前可用显存

		long long thresh = 1024 * 1024 * 1024;

		std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
		start = std::chrono::high_resolution_clock::now();

		if (avail <= get_memory_needed(image.cols, image.rows, num_of_tilt))
		{
			int col_cut = 0, row_cut = 0;
			//计算需要将图像分割几次才能不爆显存(qwq
			long long diff = avail - get_memory_needed(image.cols, image.rows, num_of_tilt);
			while (diff <= thresh)
			{
				if (col_cut == row_cut)col_cut++;
				else row_cut++;
				cudaMemGetInfo(&avail, &total);//获取当前可用显存
				diff = avail - get_memory_needed(image.cols / (col_cut + 1), image.rows / (row_cut + 1), num_of_tilt);
			}
			printf("col_cut=%d, row_cut=%d.\n", col_cut, row_cut);
			int col_beg = 0, row_beg = 0;//分割后图像范围
			int col_qtt = image.cols / (col_cut + 1);
			int row_qtt = image.rows / (row_cut + 1);
			for (int i = 0; i <= col_cut; i++)
			{
				for (int j = 0; j <= row_cut; j++)
				{
					//计算行列起点
					col_beg = i * col_qtt;
					row_beg = j * row_qtt;
					//分割图片
					cv::Rect rect(col_beg, row_beg, col_qtt, row_qtt);
					cv::Mat image_cut = cv::Mat(image, rect).clone();
					//定义关键点和描述子
					std::vector<cv::KeyPoint> kpts;
					cv::Mat des;
					//检测关键点
					DetectAndComputeAsift(image_cut, kpts, des, num_of_tilt, dopt);
					//恢复检测到的关键点的位置
					for (int k = 0, size = kpts.size(); k < size; k++)
					{
						double X = kpts[k].pt.x + i * col_qtt, Y = kpts[k].pt.y + j * row_qtt;
						double ANGLE = kpts[k].angle;
						kpts[i] = cv::KeyPoint(X, Y, -1, ANGLE);
						keypoints.push_back(kpts[i]);
					}
					descriptors.push_back(des);
				}
			}
		}
		else
		{
			DetectAndComputeAsift(image, keypoints, descriptors, num_of_tilt);
		}

		end = std::chrono::high_resolution_clock::now();
		double duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 0.000001;
		cout << "ASIFT: " << duration << " s" << endl;
	}
	void UltimateTest(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors, int detectOption, int tilt)
	{
		if (detectOption == 6)AdaptiveSegmentation(image, keypoints, descriptors, tilt, detectOption);
		else DetectAndComputeAsift(image, keypoints, descriptors, tilt, detectOption);
	}


	void SplitTest(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors)
	{
		cv::Rect rect1(0, 0, image.cols / 2, image.rows / 2);
		cv::Rect rect2(image.cols / 2, 0, image.cols / 2, image.rows / 2);
		cv::Rect rect3(0, image.rows / 2, image.cols / 2, image.rows / 2);
		cv::Rect rect4(image.cols / 2, image.rows / 2, image.cols / 2, image.rows / 2);

		cv::Mat img_1 = cv::Mat(image, rect1).clone();
		cv::Mat img_2 = cv::Mat(image, rect2).clone();
		cv::Mat img_3 = cv::Mat(image, rect3).clone();
		cv::Mat img_4 = cv::Mat(image, rect4).clone();

		std::vector<cv::KeyPoint> kpts_1, kpts_2, kpts_3, kpts_4;
		cv::Mat des_1, des_2, des_3, des_4;

		DetectAndComputeAsift(img_1, kpts_1, des_1, 7);
		for (int i = 0, size = kpts_1.size(); i < size; i++)keypoints.push_back(kpts_1[i]);
		descriptors.push_back(des_1);

		DetectAndComputeAsift(img_2, kpts_2, des_2, 7);
		for (int i = 0, size = kpts_2.size(); i < size; i++)
		{
			double X = kpts_2[i].pt.x + image.cols / 2, Y = kpts_2[i].pt.y;
			double ANGLE = kpts_2[i].angle;			
			kpts_2[i] = cv::KeyPoint(X, Y, -1, ANGLE);
			keypoints.push_back(kpts_2[i]);
		}
		descriptors.push_back(des_2);

		DetectAndComputeAsift(img_3, kpts_3, des_3, 7);
		for (int i = 0, size = kpts_3.size(); i < size; i++)
		{
			double X = kpts_3[i].pt.x, Y = kpts_3[i].pt.y + image.rows / 2;
			double ANGLE = kpts_3[i].angle;
			kpts_3[i] = cv::KeyPoint(X, Y, -1, ANGLE);
			keypoints.push_back(kpts_3[i]);
		}
		descriptors.push_back(des_3);

		DetectAndComputeAsift(img_4, kpts_4, des_4, 7);
		for (int i = 0, size = kpts_4.size(); i < size; i++)
		{
			double X = kpts_4[i].pt.x + image.cols / 2, Y = kpts_4[i].pt.y + image.rows / 2;
			double ANGLE = kpts_4[i].angle;
			kpts_4[i] = cv::KeyPoint(X, Y, -1, ANGLE);
			keypoints.push_back(kpts_4[i]);
		}
		descriptors.push_back(des_4);
	}
}
#endif