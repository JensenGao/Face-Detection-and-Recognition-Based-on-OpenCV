//////////////////////////////////////////////////////////////////////////////////////
// 收集人脸，训练人脸识别系统并识别人脸
//////////////////////////////////////////////////////////////////////////////////////


#include "recognition.h" 
#include "ImageUtils.h"  

// 收集人脸并训练；下面是三种算法
//    "FaceRecognizer.Eigenfaces":  Eigenfaces, also referred to as PCA 
//    "FaceRecognizer.Fisherfaces": Fisherfaces, also referred to as LDA
//    "FaceRecognizer.LBPH":        Local Binary Pattern Histograms 
Ptr<FaceRecognizer> learnCollectedFaces(const vector<Mat> preprocessedFaces, const vector<int> faceLabels, const string facerecAlgorithm)
{
    Ptr<FaceRecognizer> model;//建立人脸识别模型

    cout << "使用[" << facerecAlgorithm << "] 算法训练收集的人脸 ..." << endl;

    // 确保 "contrib"模块在运行时动态加载.
    bool haveContribModule = initModule_contrib();
    if (!haveContribModule) 
	{
        cerr << "ERROR: ‘contrib’模块未载入！" << endl;
        exit(1);
    }

    // 创建模型
    model = Algorithm::create<FaceRecognizer>(facerecAlgorithm);
    if (model.empty()) 
	{
        cerr << "ERROR: FaceRecognizer类中的算法 [" << facerecAlgorithm << "] 不可用！" << endl;
        exit(1);
    }

    // 训练人脸识别系统
    model->train(preprocessedFaces, faceLabels);

	//model->save("trainedModel.xml");//保存训练好的模型

    return model;
}

// 将矩阵的行或列转换为8位uchar像素的图像用于显示或保存
Mat getImageFrom1DFloatMat(const Mat matrixRow, int height)
{
    // 将一行（一个行向量）转化为图像
    Mat rectangularMat = matrixRow.reshape(1, height);
    //将图像的值设为0~255，保存为8位uchar图像
    Mat dst;
    normalize(rectangularMat, dst, 0, 255, NORM_MINMAX, CV_8UC1);
    return dst;
}

// 显示中间的人脸识别数据，可用于调试程序
void showTrainingDebugData(const Ptr<FaceRecognizer> model, const int faceWidth, const int faceHeight)
{
    try //捕捉异常
	{  
        // 算法都要先计算训练集人脸的平均值，用人脸图像减去平均人脸图像
        Mat averageFaceRow = model->get<Mat>("mean");
        printMatInfo(averageFaceRow, "averageFaceRow");
        // 将行向量转换为8位图像
        Mat averageFace = getImageFrom1DFloatMat(averageFaceRow, faceHeight);//平均脸
        printMatInfo(averageFace, "averageFace");
        imshow("averageFace", averageFace);

        // 获得特征脸
        Mat eigenvectors = model->get<Mat>("eigenvectors");
        printMatInfo(eigenvectors, "eigenvectors");

        // 显示最佳的20个特征脸
        for (int i = 0; i < min(20, eigenvectors.cols); i++) 
		{
            // 用第i个特征向量创建一个列向量
            Mat eigenvectorColumn = eigenvectors.col(i).clone();
            //printMatInfo(eigenvectorColumn, "eigenvector");

            Mat eigenface = getImageFrom1DFloatMat(eigenvectorColumn, faceHeight);//特征脸
            //printMatInfo(eigenface, "eigenface");
            imshow(format("Eigenface%d", i), eigenface);
        }

        // 获得特征值
        Mat eigenvalues = model->get<Mat>("eigenvalues");
        printMat(eigenvalues, "eigenvalues");//这里输出特征值和对应的特征向量

        //int ncomponents = model->get<int>("ncomponents");
        //cout << "ncomponents = " << ncomponents << endl;

        vector<Mat> projections = model->get<vector<Mat> >("projections");
        cout << "projections: " << projections.size() << endl;
        for (int i = 0; i < (int)projections.size(); i++) {
            printMat(projections[i], "projections");
        }

        //labels = model->get<Mat>("labels");
        //printMat(labels, "labels");

    } 
	catch (cv::Exception e) 
	{
        cout << "WARNING: 未获得FaceRecognizer的数据！" << endl;
    }
}

// 通过反向投影给定人脸的特征向量&特征值重建人脸;
//将重建人脸与要识别的人脸进行比较，求L2的相对误差，而确定识别的置信度
Mat reconstructFace(const Ptr<FaceRecognizer> model, const Mat preprocessedFace)
{
    // 因为有的算法不能重建人脸，所以这里用一个try块捕捉异常
    try 
	{
        // 从FaceRecognizer模型获得数据
        Mat eigenvectors = model->get<Mat>("eigenvectors");
        Mat averageFaceRow = model->get<Mat>("mean");

        int faceHeight = preprocessedFace.rows;

        // 将输入投影到PCA的子空间
        Mat projection = subspaceProject(eigenvectors, averageFaceRow, preprocessedFace.reshape(1,1));
        //printMatInfo(projection, "projection");

        //从PCA的子空间产生重建的人脸
        Mat reconstructionRow = subspaceReconstruct(eigenvectors, averageFaceRow, projection);
        //printMatInfo(reconstructionRow, "reconstructionRow");

        // 将行向量转换为8位像素图像，将一行（一个行向量）转化为图像
        Mat reconstructionMat = reconstructionRow.reshape(1, faceHeight);
        //将浮点像素转换为8位uchar像素
        Mat reconstructedFace = Mat(reconstructionMat.size(), CV_8U);
        reconstructionMat.convertTo(reconstructedFace, CV_8U, 1, 0);
        //printMatInfo(reconstructedFace, "reconstructedFace");

        return reconstructedFace;

    } 
	catch (cv::Exception e) 
	{
        cout << "WARNING: 未获得FaceRecognizer的数据！" << endl;
        return Mat();
    }
}
// 用L2误差比较两个图像;大于0.3（阈值）则表示两个图像不是很相似，可以采集
double getSimilarity(const Mat A, const Mat B)
{
    if (A.rows > 0 && A.rows == B.rows && A.cols > 0 && A.cols == B.cols) //这里A和B行列相等是因为检测到的人脸框用的是方形
	{
        // 计算两图像的L2误差
        double errorL2 = norm(A, B, CV_L2);
        // 平均误差：L2范数的相对误差/像素总数
        double similarity = errorL2 / (double)(A.rows * A.cols);
        return similarity;
    }
    else 
	{
        cout << "WARNING: 图像大小不一致！" << endl;
        return 100000000.0;  // 当图像不一致时返回一个很大的值
    }
}
