/*
* 这个文件用来显示Mat的结构
*/

#define USE_HIGHGUI        

#include "ImageUtils.h"

using namespace std;

//返回Mat中每个通道的位数，如: 8,16,32或64.
int getBitDepth(const cv::Mat M)
{
    switch (CV_MAT_DEPTH(M.type())) 
	{
        case CV_8U:
        case CV_8S:
            return 8;
        case CV_16U:
        case CV_16S:
            return 16;
        case CV_32S:
        case CV_32F:
            return 32;
        case CV_64F:
            return 64;
    }
    return -1;
}

//打印多通道数组的内容，方便调试(用"LOG()")
void printArray2D(const uchar *data, int cols, int rows, int channels, int depth_type, int step, int maxElements)
{
    char buff[32];
    if (data != 0 && cols > 0 && rows > 0 && channels > 0 && step > 0) 
	{

        //输出真实的数据值
        if (maxElements >= 0) 
		{
            if (maxElements == 0)
                maxElements = rows * cols;
            int totalElements = 0;
            //int step = step;

            for (int row=0; row < rows; row++) 
			{
                string s = "";
                int element = 0;
                if (channels > 1 && rows > 1) 
				{
                    snprintf(buff, sizeof(buff), "row%d: ", row);
                }
                for (int col=0; col < cols; col++) 
				{
                    if (channels > 1)
                        s += "[";
                    for (int ch=0; ch <= channels-1; ch++) 
					{
                        if (ch > 0 || (channels == 1 && col != 0))    //增加一个分离器，除此像素或行的第一个元素外
                            s += ",";

                        buff[0] = '?';  //初始化字符串
                        buff[1] = 0;

                        //打印图像的一部分
                        totalElements++;
                        if (totalElements > maxElements) 
						{
							//与printf的功能相同或类似
                            LOG("%s ... <just displaying the 1st %d entries from %d!>", s.c_str(), maxElements, rows * cols * channels);
                            return;
                        }

                        switch (depth_type) 
						{
                        case CV_8U:
                        case CV_8S:         // 8位 UCHAR Mat
                            snprintf(buff, sizeof(buff), "%d", data[(row * step) + (col*channels) + ch]);
                            break;
                        case CV_16U:
                        case CV_16S:   // 168位 short Mat.
                            snprintf(buff, sizeof(buff), "%d", *(short*)(uchar*)&data[(row * step) + ((col*channels) + ch) * sizeof(short)]);
                            break;
                        case CV_32S:   // 328位 int Mat.
                            snprintf(buff, sizeof(buff), "%d", *(int*)(uchar*)&data[(row * step) + ((col*channels) + ch) * sizeof(int)]);
                            break;
                        case CV_32F:   // 328位 float Mat.
                            snprintf(buff, sizeof(buff), "%.3f", *(float*)(uchar*)&data[(row * step) + ((col*channels) + ch) * sizeof(float)]);
                            break;
                        case CV_64F:   // 648位 double Mat.
                            snprintf(buff, sizeof(buff), "%.3lg", *(double*)(uchar*)&data[(row * step) + ((col*channels) + ch) * sizeof(double)]);
                            break;
                        default:
                            snprintf(buff, sizeof(buff), "UNKNOWN DEPTH OF %d!", depth_type);
                        }
                        s += buff;

                        const int MAX_ELEMENTS_PER_LINE = 30;   //每个LOG()声明只能打印30个数字
                        if (element > MAX_ELEMENTS_PER_LINE) 
						{
                            LOG(s.c_str());
                            s = "";
                            element = 0;
                        }
                        element++;
                    }
                    if (channels > 1)
                        s += "] ";
                }
                LOG(s.c_str());
            }
        }//end if (maxElements>=0)
    }
}

//打印图像或矩阵的内容(宽、高、通道数、每个通道位数)，方便调试代码(用"LOG()")
void printMat(const cv::Mat M, const char *label, int maxElements)
{
    string s;
    char buff[32];
    if (label)
        s = label + string(": ");
    else
        s = "Mat: ";
    if (!M.empty()) 
	{
        int channels = CV_MAT_CN(M.type());
        int depth_bpp = getBitDepth(M);     //  8, 16, 32.
        int depth_type = CV_MAT_DEPTH(M.type());    // CV_32S, CV_32F

        //打印维数和数据类型
        sprintf(buff, "%dw%dh %dch %dbpp", M.cols, M.rows, channels, depth_bpp);
        s += string(buff);

        //显示每个通道的数据范围
        s += ", range";
        for (int ch=0; ch<channels; ch++) 
		{
            cv::Mat arr = cv::Mat(M.rows, M.cols, depth_type);
            // 一次提取一个通道，并显示它的范围
            int from_to[2];
            from_to[0] = ch;
            from_to[1] = 0;
            cv::mixChannels( &M, 1, &arr, 1, from_to, 1 );
            //显示通道德范围
            double minVal, maxVal;
            cv::minMaxLoc(arr, &minVal, &maxVal);
            snprintf(buff, sizeof(buff), "[%lg,%lg]", minVal, maxVal);
            s += buff;
        }
        LOG(s.c_str());

        //显示真实的数据值（打印出来的数据）
        printArray2D(M.data, M.cols, M.rows, channels, depth_type, M.step, maxElements);
    }
    else 
	{
        LOG("%s 空Mat", s.c_str());
    }
}

//// 打印图像或矩阵的信息(宽、高、通道数、每个通道位数)，方便调试代码(用"LOG()")
void printMatInfo(const cv::Mat M, const char *label)
{
    printMat(M, label, -1);
}
