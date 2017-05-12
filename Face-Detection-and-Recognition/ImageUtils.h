/*
 * 这个文件用来显示Mat的结构
 */
#ifndef IMAGEUTILS_0_7_H_
#define IMAGEUTILS_0_7_H_

#include <cv.h>
#include <cxcore.h>
#ifdef USE_HIGHGUI
    #include <highgui.h>
#endif

#include <stdio.h>
#include <vector>
#include <string>
#include <iostream>  
#define snprintf sprintf_s   


// 这里和printf()函数功能一样，即输出函数
#ifndef LOG
    #ifndef _MSC_VER
        #define LOG(fmt, args...) do {printf(fmt, ## args); printf("\n"); fflush(stdout);} while (0)
        #define LOG printf
    #else
        #define LOG printf
    #endif
#endif


#ifdef __cplusplus
extern "C"
{
#endif

#ifdef __cplusplus
    #define DEFAULT(val) = val
#else
    #define DEFAULT(val)
#endif

// 打印图像或矩阵的内容，方便调试代码(用"LOG()")
void printMat(const cv::Mat M, const char *label DEFAULT(0), int maxElements DEFAULT(300));
// 打印图像或矩阵的信息，方便调试代码(用"LOG()")
void printMatInfo(const cv::Mat M, const char *label DEFAULT(0));

#if defined (__cplusplus)
}
#endif


#endif /* IMAGEUTILS_H_ */
