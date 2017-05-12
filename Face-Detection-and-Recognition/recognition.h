//////////////////////////////////////////////////////////////////////////////////////
// 收集人脸，训练人脸识别系统并识别人脸
//////////////////////////////////////////////////////////////////////////////////////

#pragma once
#include <stdio.h>
#include <iostream>
#include <vector>

#include "opencv2/opencv.hpp"
using namespace cv;
using namespace std;



// 收集人脸并训练；下面是三种算法
//    "FaceRecognizer.Eigenfaces":  Eigenfaces, also referred to as PCA 
//    "FaceRecognizer.Fisherfaces": Fisherfaces, also referred to as LDA
//    "FaceRecognizer.LBPH":        Local Binary Pattern Histograms 
Ptr<FaceRecognizer> learnCollectedFaces(const vector<Mat> preprocessedFaces, const vector<int> faceLabels, const string facerecAlgorithm = "FaceRecognizer.Eigenfaces");

// 显示中间的人脸识别数据，可用于调试程序
void showTrainingDebugData(const Ptr<FaceRecognizer> model, const int faceWidth, const int faceHeight);

// 通过反向投影给定人脸的特征向量&特征值重建人脸
Mat reconstructFace(const Ptr<FaceRecognizer> model, const Mat preprocessedFace);

// 用L2误差比较两个图像
double getSimilarity(const Mat A, const Mat B);
