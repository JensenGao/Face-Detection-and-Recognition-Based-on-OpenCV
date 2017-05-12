//////////////////////////////////////////////////////////////////////////////////////
// 人脸识别项目主函数（包含GUI的制作）
//////////////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <vector>
#include <string>
#include <iostream>

#include "opencv2/opencv.hpp"

#include "detectObject.h"       
#include "preprocessFace.h"    
#include "recognition.h"    
#include "ImageUtils.h"    

using namespace cv;
using namespace std;


#if !defined VK_ESCAPE
    #define VK_ESCAPE 0x1B      //按“Esc”键退出系统
#endif

// 人脸识别算法，下面是三种算法：
//    "FaceRecognizer.Eigenfaces":  Eigenfaces, also referred to as PCA
//    "FaceRecognizer.Fisherfaces": Fisherfaces, also referred to as LDA 
//    "FaceRecognizer.LBPH":        Local Binary Pattern Histograms
const char *facerecAlgorithm = "FaceRecognizer.Fisherfaces";//直接修改这里就可以改变算法
//const char *facerecAlgorithm = "FaceRecognizer.Eigenfaces";


//置信度阈值：Eigenfaces 0.5、 Fisherfaces 0.7；这里的阈值是判断一个人是否在训练集内的阈值，当大于该阈值时，表示是一个已知的人脸
//相似度阈值=1-置信度阈值；相似度阈值：getSimilarity返回的L2误差平均误差
const float UNKNOWN_PERSON_THRESHOLD = 0.7f;

// 级联分类器
const char *faceCascadeFilename = "C:/opencv/sources/data/lbpcascades/lbpcascade_frontalface.xml";     // LBP face detector.
//const char *faceCascadeFilename = "C:/opencv/sources/data/lbpcascades/haarcascade_frontalface_alt_tree.xml";  // Haar face detector.
//const char *eyeCascadeFilename1 = "C:/opencv/sources/data/lbpcascades/haarcascade_lefteye_2splits.xml";   // Best eye detector for open-or-closed eyes.
//const char *eyeCascadeFilename2 = "C:/opencv/sources/data/lbpcascades/haarcascade_righteye_2splits.xml";   // Best eye detector for open-or-closed eyes.
//const char *eyeCascadeFilename1 = "C:/opencv/sources/data/lbpcascades/haarcascade_mcs_lefteye.xml";       // Good eye detector for open-or-closed eyes.
//const char *eyeCascadeFilename2 = "C:/opencv/sources/data/lbpcascades/haarcascade_mcs_righteye.xml";       // Good eye detector for open-or-closed eyes.
const char *eyeCascadeFilename1 = "C:/opencv/sources/data/haarcascades/haarcascade_eye.xml";               // Basic eye detector for open eyes only.
const char *eyeCascadeFilename2 = "C:/opencv/sources/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml"; // Basic eye detector for open eyes if they might wear glasses.


// 设置期望的人脸维度，设置为70*70
const int faceWidth = 70;
const int faceHeight = faceWidth;

//设置摄像机分辨率
const int DESIRED_CAMERA_WIDTH = 640;
const int DESIRED_CAMERA_HEIGHT = 480;

//人脸图像采集间隔，间隔太小则视为同一人脸图像
const double CHANGE_IN_IMAGE_FOR_COLLECTION = 0.3;      //getSimilarity返回的两图像相似度的平均误差的阈值，当大于该值时表示两图像不是很相似，可以采集
const double CHANGE_IN_SECONDS_FOR_COLLECTION = 1.0;       //间隔1s采集一次人脸图像

const char *windowName = "WebcamFaceRec";   // GUI的名字
const int BORDER = 8;  // Border between GUI elements to the edge of the image.

const bool preprocessLeftAndRightSeparately = true;   // 是否分别对左侧和右侧人脸进行处理的标志

// Set to true if you want to see many windows created, showing various debug info. Set to 0 otherwise.
bool m_debug = false;

// 系统交互式GUI的运行模式
enum MODES {MODE_STARTUP=0, MODE_DETECTION, MODE_COLLECT_FACES, MODE_TRAINING, MODE_RECOGNITION, MODE_DELETE_ALL,   MODE_END};
const char* MODE_NAMES[] = {"Startup", "Detection", "Collect Faces", "Training", "Recognition", "Delete All", "ERROR!"};
MODES m_mode = MODE_STARTUP;

int m_selectedPerson = -1;//总人数
int m_numPersons = 0;
vector<int> m_latestFaces;//最新收集的人脸图像

//GUI中按键的位置
Rect m_rcBtnAdd;
Rect m_rcBtnDel;
Rect m_rcBtnDebug;
int m_gui_faces_left = -1;
int m_gui_faces_top = -1;

// 用于将整形或浮点型与string型相互转换
template <typename T> string toString(T t)
{
    ostringstream out;
    out << t;
    return out.str();
}

template <typename T> T fromString(string t)
{
    T out;
    istringstream in(t);
    in >> out;
    return out;
}

// 加载人脸和左眼、右眼的检测器
void initDetectors(CascadeClassifier &faceCascade, CascadeClassifier &eyeCascade1, CascadeClassifier &eyeCascade2)
{
    // 载入人脸检测级联分类器-xml文件
    try 
	{   
        faceCascade.load(faceCascadeFilename);
    } 
	catch (cv::Exception &e) 
	{ }
    if ( faceCascade.empty() ) 
	{
        cerr << "ERROR: 载入人脸检测级联分类器[" << faceCascadeFilename << "]失败!" << endl;
        exit(1);
    }
    cout << "载入人脸检测级联分类器[" << faceCascadeFilename << "]成功。" << endl;

    //// 载入眼睛检测级联分类器-xml文件 
    try {   
        eyeCascade1.load(eyeCascadeFilename1);
    } 
	catch (cv::Exception &e) 
	{}
    if ( eyeCascade1.empty() ) 
	{
        cerr << "ERROR:载入第一个眼睛检测级联分类器[" << eyeCascadeFilename1 << "]失败!" << endl;
       exit(1);
    }
    cout << "载入第一个眼睛检测级联分类器[" << eyeCascadeFilename1 << "]成功。" << endl;

    try {  
        eyeCascade2.load(eyeCascadeFilename2);
    } 
	catch (cv::Exception &e) 
	{}
    if ( eyeCascade2.empty() ) 
	{
        cerr << "ERROR:载入第二个眼睛检测级联分类器[" << eyeCascadeFilename2 << "]失败！" << endl;
        exit(1);
    }
    else
        cout << "载入第二个眼睛检测级联分类器[" << eyeCascadeFilename2 << "]成功。" << endl;
}

// 打开摄像机
void initWebcam(VideoCapture &videoCapture, int cameraNumber)
{
    try {   
        videoCapture.open(cameraNumber);
    } 
	catch (cv::Exception &e) 
	{}
    if ( !videoCapture.isOpened() ) 
	{
        cerr << "ERROR: 打开摄像机失败！" << endl;
        exit(1);
    }
    cout << "载入摄像机编号为：" << cameraNumber << "." << endl;
}


// 在图像中加入文本
Rect drawString(Mat img, string text, Point coord, Scalar color, float fontScale = 0.6f, int thickness = 1, int fontFace = FONT_HERSHEY_COMPLEX)
{
    //获取文本大小和位置
    int baseline=0;
    Size textSize = getTextSize(text, fontFace, fontScale, thickness, &baseline);
    baseline += thickness;

    // 调整文本位置：左右或上下调整
    if (coord.y >= 0) 
	{
        //将位于图像左上角的文本下移
        coord.y += textSize.height;
    }
    else 
	{
        //坐标位于图像左下角的文本上移
        coord.y += img.rows - baseline + 1;
    }
    // 向右调整
    if (coord.x < 0) 
	{
        coord.x += img.cols - textSize.width + 1;
    }

    //获得文本边界
    Rect boundingRect = Rect(coord.x, coord.y - textSize.height, textSize.width, baseline + textSize.height);

    // Draw anti-aliased text.
    cv::putText(img, text, coord, fontFace, fontScale, color, thickness, CV_AA);

    // 返回文本边界矩形
    return boundingRect;
}

// 绘制GUI上的按钮
Rect drawButton(Mat img, string text, Point coord, int minWidth = 0)
{
    int B = BORDER;
    Point textCoord = Point(coord.x + B, coord.y + B);
    // 获得文本边界矩形
    Rect rcText = drawString(img, text, textCoord, CV_RGB(0,0,0));
    // 在文本周围绘制一个填充矩形
    Rect rcButton = Rect(rcText.x - B, rcText.y - B, rcText.width + 2*B, rcText.height + 2*B);
    // 设置最小按钮宽度
    if (rcButton.width < minWidth)
        rcButton.width = minWidth;
    // 半透明白色矩形
    Mat matButton = img(rcButton);
    matButton += CV_RGB(90, 90, 90);
    //绘制不透明的白色边界
    rectangle(img, rcButton, CV_RGB(200,200,200), 1, CV_AA);

    //绘制将显示的文本
    drawString(img, text, textCoord, CV_RGB(10,55,20));

    return rcButton;
}

//判断鼠标点击位置是否在按钮区域（也可以用opencv中的inside函数来判断-pt.inside）
bool isPointInRect(const Point pt, const Rect rc)
{
    if (pt.x >= rc.x && pt.x <= (rc.x + rc.width - 1))
        if (pt.y >= rc.y && pt.y <= (rc.y + rc.height - 1))
            return true;

    return false;
}

// 鼠标事件：当点击GUI窗口时调用本函数
void onMouse(int event, int x, int y, int, void*)
{
    // 只关心鼠标左键的DOWN事件
    if (event != CV_EVENT_LBUTTONDOWN)
        return;

    // 判断用户是否点击GUI按钮
    Point pt = Point(x,y);
    if (isPointInRect(pt, m_rcBtnAdd)) 
	{
        cout << "用户点击[Add Person]按钮，此时人数为" << m_numPersons << endl;
        // 当检测到一个人脸，但未收集该人脸时，则收集
        if ((m_numPersons == 0) || (m_latestFaces[m_numPersons-1] >= 0)) 
		{
            // 添加一个新人（人数+1）
            m_numPersons++;
            m_latestFaces.push_back(-1); // 为新人分配空间
            cout << "人数为: " << m_numPersons << endl;
        }
        // 使用新添加进来的人
        m_selectedPerson = m_numPersons - 1;
        m_mode = MODE_COLLECT_FACES;
    }
    else if (isPointInRect(pt, m_rcBtnDel)) 
	{
        cout << "用户点击[Delete All]按钮。" << endl;
        m_mode = MODE_DELETE_ALL;
    }
    else if (isPointInRect(pt, m_rcBtnDebug)) 
	{
        cout << "用户点击 [Debug] 按钮。" << endl;
        m_debug = !m_debug;
        cout << "Debug 模式: " << m_debug << endl;
    }
    else 
	{
        cout << "用户点击图像。" << endl;
        //检测用户是否点击了已存在于列表中的人脸 
        int clickedPerson = -1;
        for (int i=0; i<m_numPersons; i++) 
		{
            if (m_gui_faces_top >= 0) 
			{
                Rect rcFace = Rect(m_gui_faces_left, m_gui_faces_top + i * faceHeight, faceWidth, faceHeight);
                if (isPointInRect(pt, rcFace)) 
				{
                    clickedPerson = i;
                    break;
                }
            }
        }
        // 用户点击GUI中的人脸时，改变已经选择的人
        if (clickedPerson >= 0) 
		{
            // 改变当前的人并收集当前人的人脸（这里就是换了一个人的意思）
            m_selectedPerson = clickedPerson; // 使用新添加的人的人脸
            m_mode = MODE_COLLECT_FACES;
        }
        // 这里表示点击了GUI的中心
        else 
		{
            // 已经收集人脸时转换至训练模式
            if (m_mode == MODE_COLLECT_FACES) 
			{
                cout << "用户希望开始训练！" << endl;
                m_mode = MODE_TRAINING;
            }
        }
    }
}


//这个函数只有在检测到“Esc”键时才会退出
void recognizeAndTrainUsingWebcam(VideoCapture &videoCapture, CascadeClassifier &faceCascade, CascadeClassifier &eyeCascade1, CascadeClassifier &eyeCascade2)
{
    Ptr<FaceRecognizer> model;
    vector<Mat> preprocessedFaces;
    vector<int> faceLabels;
    Mat old_prepreprocessedFace;
    double old_time = 0;

    //进入检测模式
    m_mode = MODE_DETECTION;

    //循环只有在检测到“Esc”键值时才会退出
    while (true) {

        //读取摄像机采集的图像
        Mat cameraFrame;
        videoCapture >> cameraFrame;
        if( cameraFrame.empty() ) {
            cerr << "ERROR: 读取图像失败！" << endl;
            exit(1);
        }

        //复制图像用于绘制
        Mat displayedFrame;
        cameraFrame.copyTo(displayedFrame);

        //这个变量是表示识别出来的人的标签
        int identity = -1;

        //检测到人脸并进行预处理（需要标准大小，对比度和亮度）
        Rect faceRect;  // 检测出来的人脸的位置
        Rect searchedLeftEye, searchedRightEye; //左右眼检测
        Point leftEye, rightEye;    // 标记检测出来的眼睛
        Mat preprocessedFace = getPreprocessedFace(displayedFrame, faceWidth, faceCascade, eyeCascade1, eyeCascade2, preprocessLeftAndRightSeparately, &faceRect, &leftEye, &rightEye, &searchedLeftEye, &searchedRightEye);

        bool gotFaceAndEyes = false;
        if (preprocessedFace.data)
            gotFaceAndEyes = true;

        // 在检测到的人脸周围绘制一个矩形
        if (faceRect.width > 0) 
		{
            rectangle(displayedFrame, faceRect, CV_RGB(255, 255, 0), 2, CV_AA);

            // 用蓝线画出眼睛的位置
            Scalar eyeColor = CV_RGB(0,255,255);
            if (leftEye.x >= 0) //判断是否检测到左眼
			{   
                circle(displayedFrame, Point(faceRect.x + leftEye.x, faceRect.y + leftEye.y), 6, eyeColor, 1, CV_AA);
            }
            if (rightEye.x >= 0) 
			{   
                circle(displayedFrame, Point(faceRect.x + rightEye.x, faceRect.y + rightEye.y), 6, eyeColor, 1, CV_AA);
            }
        }

        if (m_mode == MODE_DETECTION) 
		{
            
        }
        else if (m_mode == MODE_COLLECT_FACES) 
		{
            // 判断是否检测到人脸
            if (gotFaceAndEyes) 
			{

                //将检测到人脸与先前收集的人脸进行比较：L2误差
                double imageDiff = 10000000000.0;
                if (old_prepreprocessedFace.data) 
				{
                    imageDiff = getSimilarity(preprocessedFace, old_prepreprocessedFace);
			    }

				//采集预处理后的人脸图像的时间间隔
                double current_time = (double)getTickCount();
                double timeDiff_seconds = (current_time - old_time)/getTickFrequency();

                // 只有在检测到的人脸与上一帧的人脸明显不同并且有一定的时间间隔时才进行处理
                if ((imageDiff > CHANGE_IN_IMAGE_FOR_COLLECTION) && (timeDiff_seconds > CHANGE_IN_SECONDS_FOR_COLLECTION)) 
				{
                    //数据增强：镜像
                    Mat mirroredFace;
                    flip(preprocessedFace, mirroredFace, 1);

                    // 将人脸图像加入已检测到的人脸列表
                    preprocessedFaces.push_back(preprocessedFace);//检测到的人脸加入，作为训练集
                    preprocessedFaces.push_back(mirroredFace);
                    faceLabels.push_back(m_selectedPerson);//标签，原图像和镜像图像都要加标签，所以有两次
                    faceLabels.push_back(m_selectedPerson);

                    //m_latestFaces保存每个人在大数组preprocessedFaces（存储原图像和镜像图像，因此引用倒数第二个人脸）的引用					
                    m_latestFaces[m_selectedPerson] = preprocessedFaces.size() - 2; //指向非镜像人脸（减1是最后一个镜像人脸，减2就是非镜像人脸）
                    cout << "保存了 " << (preprocessedFaces.size()/2) << "张人脸图像，总共" << m_selectedPerson<<"人。" << endl;

                    //当采集人脸图像时，GUI中的人脸会闪一下（显示彩色视频帧的复制）
                    Mat displayedFaceRegion = displayedFrame(faceRect);
                    displayedFaceRegion += CV_RGB(90,90,90);

                    // 保存处理好的人脸用于下一次迭代
                    old_prepreprocessedFace = preprocessedFace;
                    old_time = current_time;
                }
            }
        }
        else if (m_mode == MODE_TRAINING)//训练模式
		{
            //判断是否有足够的数据进行训练（Eigenfaces：一个人的人脸就行；Fisherfaces：至少要两个人，否则程序会崩溃）
            bool haveEnoughData = true;
            if (strcmp(facerecAlgorithm, "FaceRecognizer.Fisherfaces") == 0) 
			{
                if ((m_numPersons < 2) || (m_numPersons == 2 && m_latestFaces[1] < 0) ) 
				{
                    cout << "Warning: Fisherfaces至少需要两个人的人脸才能开始训练！请收集更多数据..." << endl;
                    haveEnoughData = false;
                }
            }
            if (m_numPersons < 1 || preprocessedFaces.size() <= 0 || preprocessedFaces.size() != faceLabels.size()) 
			{
                cout << "Warning:在学习之前需要训练数据！ 请收集更多数据..." << endl;
                haveEnoughData = false;
            }

            if (haveEnoughData) {
                // 开始训练
                model = learnCollectedFaces(preprocessedFaces, faceLabels, facerecAlgorithm);

                // 显示人脸识别的中间过程数据，可以帮助调试
                if (m_debug)
                    showTrainingDebugData(model, faceWidth, faceHeight);

                // 训练完成，可以进行识别
                m_mode = MODE_RECOGNITION;
            }
            else 
			{
                //数据不够时不能进行训练，而要继续收集人脸
                m_mode = MODE_COLLECT_FACES;
            }

        }
        else if (m_mode == MODE_RECOGNITION) //识别模式
		{
            if (gotFaceAndEyes && (preprocessedFaces.size() > 0) && (preprocessedFaces.size() == faceLabels.size())) 
			{
                // 反向投影特征向量和特征值重建人脸
                Mat reconstructedFace;
                reconstructedFace = reconstructFace(model, preprocessedFace);
                if (m_debug)
                    if (reconstructedFace.data)
                        imshow("reconstructedFace", reconstructedFace);

                //验证重建的人脸是否和预处理的人脸相似，从而判断是否是一个不认识的人（训练集中没有）
                double similarity = getSimilarity(preprocessedFace, reconstructedFace);

                string outputStr;
                if (similarity < UNKNOWN_PERSON_THRESHOLD) 
				{
                    // 进行识别，返回人的标签，即是哪个人
                    identity = model->predict(preprocessedFace);
                    outputStr = toString(identity);
                }
                else 
				{
                    // 当置信度小于阈值时，则为不认识的人
                    outputStr = "Unknown";
                }
                cout << "Identity: " << outputStr << "。 Similarity: " << similarity << endl;

                //显示置信度的位置
                int cx = (displayedFrame.cols - faceWidth) / 2;
                Point ptBottomRight = Point(cx - 5, BORDER + faceHeight);
                Point ptTopLeft = Point(cx - 15, BORDER);
                // 当识别的人是"unknown"时，则用灰色标记阈值
                Point ptThreshold = Point(ptTopLeft.x, ptBottomRight.y - (1.0 - UNKNOWN_PERSON_THRESHOLD) * faceHeight);
                rectangle(displayedFrame, ptThreshold, Point(ptBottomRight.x, ptThreshold.y), CV_RGB(200,200,200), 1, CV_AA);
                // 将置信度转换为0.0 ~1.0 之间
                double confidenceRatio = 1.0 - min(max(similarity, 0.0), 1.0);
                Point ptConfidence = Point(ptTopLeft.x, ptBottomRight.y - confidenceRatio * faceHeight);
                //显示置信度
                rectangle(displayedFrame, ptConfidence, ptBottomRight, CV_RGB(0,255,255), CV_FILLED, CV_AA);
                //显示灰色置信度边界
                rectangle(displayedFrame, ptTopLeft, ptBottomRight, CV_RGB(200,200,200), 1, CV_AA);
            }
        }
        else if (m_mode == MODE_DELETE_ALL) //清空
		{
            // 初始化所有变量
            m_selectedPerson = -1;
            m_numPersons = 0;
            m_latestFaces.clear();
            preprocessedFaces.clear();
            faceLabels.clear();
            old_prepreprocessedFace = Mat();

            //从检测模式开始
            m_mode = MODE_DETECTION;
        }
        else 
		{
            cerr << "ERROR: 无效的模式！" << m_mode << endl;
            exit(1);
        }
		        
        // 显示帮助,同时也显示出收集的人脸数（不包括镜像人脸）
        string help;
        Rect rcHelp;
        if (m_mode == MODE_DETECTION)
            help = "检测模式：点击 [Add Person]按钮准备收集人脸。";
        else if (m_mode == MODE_COLLECT_FACES)
			help = "收集模式：点击任何地方开始训练收集的" + toString(m_numPersons) + "个人的" + toString(preprocessedFaces.size() / 2) + " 张人脸图像！ ";
        else if (m_mode == MODE_TRAINING)
			help = "训练模式：当在训练" + toString(m_numPersons) + "个人的" + toString(preprocessedFaces.size() / 2) + " 张人脸图像时，请等待！";
        else if (m_mode == MODE_RECOGNITION)
            help = "识别模式：点击GUI右边的人的人脸，为这个人添加更多的数据，或者点击 [Add Person] 添加新人！";
        if (help.length() > 0) 
		{
            // 显示“help”：黑底，白字
            float txtSize = 0.4;
            drawString(displayedFrame, help, Point(BORDER, -BORDER-2), CV_RGB(0,0,0), txtSize);  //黑色阴影
            rcHelp = drawString(displayedFrame, help, Point(BORDER+1, -BORDER-1), CV_RGB(255,255,255), txtSize);  //白色文本
        }

        // 显示当前的模式
        if (m_mode >= 0 && m_mode < MODE_END) 
		{
            string modeStr = "MODE: " + string(MODE_NAMES[m_mode]);
            drawString(displayedFrame, modeStr, Point(BORDER, -BORDER-2 - rcHelp.height), CV_RGB(0,0,0));      
            drawString(displayedFrame, modeStr, Point(BORDER+1, -BORDER-1 - rcHelp.height), CV_RGB(0,255,0)); // 绿色文本
        }

        // 在GUI顶部中心显示当前预处理的人脸
        int cx = (displayedFrame.cols - faceWidth) / 2;
        if (preprocessedFace.data) 
		{
            // 灰度图转换为BGR图像
            Mat srcBGR = Mat(preprocessedFace.size(), CV_8UC3);
            cvtColor(preprocessedFace, srcBGR, CV_GRAY2BGR);
            // 获得需要的ROI
            Rect dstRC = Rect(cx, BORDER, faceWidth, faceHeight);
            Mat dstROI = displayedFrame(dstRC);
            srcBGR.copyTo(dstROI);
        }
        //在人脸周围绘制一个边界
        rectangle(displayedFrame, Rect(cx-1, BORDER-1, faceWidth+2, faceHeight+2), CV_RGB(200,200,200), 1, CV_AA);

        //在图像中绘制GUI按钮
        m_rcBtnAdd = drawButton(displayedFrame, "Add Person", Point(BORDER, BORDER));
        m_rcBtnDel = drawButton(displayedFrame, "Delete All", Point(m_rcBtnAdd.x, m_rcBtnAdd.y + m_rcBtnAdd.height), m_rcBtnAdd.width);
        m_rcBtnDebug = drawButton(displayedFrame, "Debug", Point(m_rcBtnDel.x, m_rcBtnDel.y + m_rcBtnDel.height), m_rcBtnAdd.width);

        // 在右侧显示为每个人最新收集的人脸（所有人都显示）
        m_gui_faces_left = displayedFrame.cols - BORDER - faceWidth;
        m_gui_faces_top = BORDER;
        for (int i=0; i<m_numPersons; i++) 
		{
            int index = m_latestFaces[i];
            if (index >= 0 && index < (int)preprocessedFaces.size()) 
			{
                Mat srcGray = preprocessedFaces[index];
                if (srcGray.data) 
				{
                    // 灰度图转换为BGR图像
                    Mat srcBGR = Mat(srcGray.size(), CV_8UC3);
                    cvtColor(srcGray, srcBGR, CV_GRAY2BGR);
                    //// 获得需要的ROI
                    int y = min(m_gui_faces_top + i * faceHeight, displayedFrame.rows - faceHeight);
                    Rect dstRC = Rect(m_gui_faces_left, y, faceWidth, faceHeight);
                    Mat dstROI = displayedFrame(dstRC);
                    srcBGR.copyTo(dstROI);
                }
            }
        }

        //突出正在收集的人的人脸，在人脸周围使用红色矩形
        if (m_mode == MODE_COLLECT_FACES) 
		{
            if (m_selectedPerson >= 0 && m_selectedPerson < m_numPersons) 
			{
                int y = min(m_gui_faces_top + m_selectedPerson * faceHeight, displayedFrame.rows - faceHeight);
                Rect rc = Rect(m_gui_faces_left, y, faceWidth, faceHeight);
                rectangle(displayedFrame, rc, CV_RGB(255,0,0), 3, CV_AA);
            }
        }

        //突出要识别的人，在人脸周围用绿色矩形
        if (identity >= 0 && identity < 1000) 
		{
            int y = min(m_gui_faces_top + identity * faceHeight, displayedFrame.rows - faceHeight);
            Rect rc = Rect(m_gui_faces_left, y, faceWidth, faceHeight);
            rectangle(displayedFrame, rc, CV_RGB(0,255,0), 3, CV_AA);
        }

        //显示摄像机的图像
        imshow(windowName, displayedFrame);

        // 当需要调试数据时，显示中间过程的数据
        if (m_debug) 
		{
            Mat face;
            if (faceRect.width > 0) 
			{
                face = cameraFrame(faceRect);
                if (searchedLeftEye.width > 0 && searchedRightEye.width > 0) 
				{
                    Mat topLeftOfFace = face(searchedLeftEye);
                    Mat topRightOfFace = face(searchedRightEye);
                    imshow("topLeftOfFace", topLeftOfFace);
                    imshow("topRightOfFace", topRightOfFace);
                }
            }

            if (!model.empty())
                showTrainingDebugData(model, faceWidth, faceHeight);
        }

        
        //检测按键
        char keypress = waitKey(20); 

        if (keypress == VK_ESCAPE) //Esc
		{ 
            break;
        }

    }//end while
}

int main(int argc, char *argv[])
{
    CascadeClassifier faceCascade;
    CascadeClassifier eyeCascade1;
    CascadeClassifier eyeCascade2;
    VideoCapture videoCapture;

    //载入XML分类器
    initDetectors(faceCascade, eyeCascade1, eyeCascade2);

    cout << endl;
    cout << "按'Esc'键退出GUI!" << endl;

    
    int cameraNumber = 0;   // 指定摄像机
    if (argc > 1) 
	{
        cameraNumber = atoi(argv[1]);
    }

    // 访问相机
    initWebcam(videoCapture, cameraNumber);

    // 设置帧的分辨率：640*480
    videoCapture.set(CV_CAP_PROP_FRAME_WIDTH, DESIRED_CAMERA_WIDTH);
    videoCapture.set(CV_CAP_PROP_FRAME_HEIGHT, DESIRED_CAMERA_HEIGHT);

    //创建窗口显示图像并用于创建GUI
    namedWindow(windowName);
    //捕捉GUI窗口中的鼠标事件
    setMouseCallback(windowName, onMouse, 0);

    // 运行人脸识别系统
    recognizeAndTrainUsingWebcam(videoCapture, faceCascade, eyeCascade1, eyeCascade2);

    return 0;
}
