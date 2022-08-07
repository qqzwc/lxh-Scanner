/*
部分参考：https://blog.csdn.net/RayChiu757374816/article/details/120221451
一般的图像文件格式使用的是 Unsigned 8位，CvMat矩阵对应的参数类型就是
CV_8UC1，CV_8UC2，CV_8UC3。（最后的1、2、3表示通道数，譬如RGB3通道就用CV_8UC3）
float是32位的，对应CvMat数据结构参数就是：CV_32FC1，CV_32FC2，CV_32FC3...
double是64位的，对应CvMat数据结构参数：CV_64FC1，CV_64FC2，CV_64FC3等。
*/
#pragma once
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "lxh_grabcut.hpp"

#ifdef LXH_OUTPUT
fstream fAppLog("./output/lxh_app.log", ios::out);
#endif

lxh::Timer appTimer;
const cv::Scalar RED = cv::Scalar(0, 0, 255); // 前景画笔的颜色
const cv::Scalar PINK = cv::Scalar(230, 130, 255); // 可能的前景画笔颜色
const cv::Scalar BLUE = cv::Scalar(255, 0, 0); // 背景画笔的颜色
const cv::Scalar LIGHTBLUE = cv::Scalar(255, 255, 160); // 可能的背景画笔颜色
const cv::Scalar GREEN = cv::Scalar(0, 255, 0); // 矩形框颜色
const int FGD_KEY = cv::EVENT_FLAG_SHIFTKEY; // 按下SHIFT表示使用前景画笔
const int BGD_KEY = cv::EVENT_FLAG_CTRLKEY; // 按下CTRL表示使用背景画笔

void makeBinMask(const cv::Mat& comMask, cv::Mat& binMask)
{
    if (comMask.empty() || comMask.type() != CV_8UC1)
        CV_Error(cv::Error::StsBadArg, "comMask is empty or has incorrect type (not CV_8UC1)");
    if (binMask.empty() || binMask.rows != comMask.rows || binMask.cols != comMask.cols)
        binMask.create(comMask.size(), CV_8UC1);
    binMask = comMask & 1; // 只有背景是黑的
}
class UIApp {
public:
    enum { NOT_SET = 0,
        IN_PROCESS = 1,
        SET = 2 };
    static const int radius = 2;
    static const int thickness = -1;
    void reset();
    void setImageAndWinName(const cv::Mat& _img, const string& _winName);
    void showImage() const;
    void mouseClick(int event, int x, int y, int flags, void* param);
    int runGrabCut(int iterCount, lxh::GMMInitType initType, bool useRefine=false);
    int runGraphCut();
    int getIterCount() const { return nowIterCount; }
    UIApp(double _gamma=50){
        winName = nullptr;
        img = nullptr;
        grabCut = nullptr;
        gamma = _gamma;
    }
    ~UIApp(){
        if (grabCut != nullptr)
            delete grabCut;
    }
    cv::Mat getBinMask(){
        cv::Mat binMask;
        makeBinMask(mask, binMask);
        return binMask; // 只返回矩形区域
    }
    cv::Rect getRect(){
        return rect;
    }
    
private:
    void setRectInMask();
    void setLabelInMask(int flags, cv::Point p, bool isPr);
    const string* winName;
    const cv::Mat* img;
    lxh::GrabCut2D* grabCut;
    cv::Mat mask;
    cv::Mat bgdModel, fgdModel;
    uchar rectState, labelState, prLabelState;
    bool isInitialized;
    cv::Rect rect;
    vector<cv::Point> fgdPxls, bgdPxls, prFgdPxls, prBgdPxls;
    int nowIterCount;
    double gamma;
};
void UIApp::reset()
{
    if (!mask.empty())
        mask.setTo(cv::Scalar::all(lxh::BGD));
    bgdPxls.clear();
    fgdPxls.clear();
    prBgdPxls.clear();
    prFgdPxls.clear();
    isInitialized = false;
    rectState = NOT_SET;
    labelState = NOT_SET;
    prLabelState = NOT_SET;
    nowIterCount = 0;
    if (grabCut != nullptr){
        delete grabCut;
        grabCut = nullptr;
    }
    if (img != nullptr)
        grabCut = new lxh::GrabCut2D(*img, gamma);
}
void UIApp::setImageAndWinName(const cv::Mat& _img, const string& _winName)
{
    if (_img.empty() || _winName.empty())
        return;
    img = &_img;
    winName = &_winName;
    mask.create(img->size(), CV_8UC1);
    reset(); // 在reset里面初始化grabcut
}
void UIApp::showImage() const
{
    if (img->empty() || winName->empty())
        return;
    cv::Mat res;
    cv::Mat binMask;
    if (!isInitialized)
        img->copyTo(res);
    else { // 未初始化时加入掩膜
        makeBinMask(mask, binMask); // 除了确定背景其他全当前景显示
        img->copyTo(res, binMask);
    }
    vector<cv::Point>::const_iterator it;
    for (it = bgdPxls.begin(); it != bgdPxls.end(); ++it)
        circle(res, *it, radius, BLUE, thickness);
    for (it = fgdPxls.begin(); it != fgdPxls.end(); ++it)
        circle(res, *it, radius, RED, thickness);
    for (it = prBgdPxls.begin(); it != prBgdPxls.end(); ++it)
        circle(res, *it, radius, LIGHTBLUE, thickness);
    for (it = prFgdPxls.begin(); it != prFgdPxls.end(); ++it)
        circle(res, *it, radius, PINK, thickness);
    if (rectState == IN_PROCESS || rectState == SET)
        rectangle(res, cv::Point(rect.x, rect.y), cv::Point(rect.x + rect.width, rect.y + rect.height), GREEN, 2);
    imshow(*winName, res);
}
void UIApp::setRectInMask()
{
    CV_Assert(!mask.empty());
    mask.setTo(lxh::BGD);
    rect.x = max(0, rect.x);
    rect.y = max(0, rect.y);
    rect.width = min(rect.width, img->cols - rect.x);
    rect.height = min(rect.height, img->rows - rect.y);
    (mask(rect)).setTo(cv::Scalar(lxh::PR_FGD));
}
void UIApp::setLabelInMask(int flags, cv::Point p, bool isPr)
{ // 添加鼠标拖动的label
    vector<cv::Point> *bpxls, *fpxls;
    uchar bvalue, fvalue;
    if (!isPr) {
        bpxls = &bgdPxls;
        fpxls = &fgdPxls;
        bvalue = lxh::BGD;
        fvalue = lxh::FGD;
    } else {
        bpxls = &prBgdPxls;
        fpxls = &prFgdPxls;
        bvalue = lxh::PR_BGD;
        fvalue = lxh::PR_FGD;
    }
    if (flags & BGD_KEY) {
        bpxls->push_back(p);
        circle(mask, p, radius, bvalue, thickness);
    }
    if (flags & FGD_KEY) {
        fpxls->push_back(p);
        circle(mask, p, radius, fvalue, thickness);
    }
}
void UIApp::mouseClick(int event, int x, int y, int flags, void*)
{
    // TODO add bad args check
    switch (event) {
    case cv::EVENT_LBUTTONDOWN: // set rect or GC_BGD(GC_FGD) labels 鼠标左键按下
    {
        bool isB = (flags & BGD_KEY) != 0, isF = (flags & FGD_KEY) != 0;
        if (rectState == NOT_SET && !isB && !isF) {
            rectState = IN_PROCESS;
            rect = cv::Rect(x, y, 1, 1); // 记录矩形左上角
        }
        if ((isB || isF) && rectState == SET)
            labelState = IN_PROCESS; // 开始记录用户给定的种子
    } break;
    case cv::EVENT_RBUTTONDOWN: // set GC_PR_BGD(GC_PR_FGD) labels 鼠标右键按下
    {
        bool isB = (flags & BGD_KEY) != 0,
             isF = (flags & FGD_KEY) != 0;
        if ((isB || isF) && rectState == SET)
            prLabelState = IN_PROCESS;
    } break;
    case cv::EVENT_LBUTTONUP: // 鼠标左键松开
        if (rectState == IN_PROCESS) { 
            rect = cv::Rect(cv::Point(rect.x, rect.y), cv::Point(x, y)); // 记录矩形右上角，至此完成一个完整矩形
            rectState = SET;
            setRectInMask();
            CV_Assert(bgdPxls.empty() && fgdPxls.empty() && prBgdPxls.empty() && prFgdPxls.empty());
            showImage();
        }
        if (labelState == IN_PROCESS) {
            setLabelInMask(flags, cv::Point(x, y), false);
            labelState = SET;
            showImage();
        }
        break;
    case cv::EVENT_RBUTTONUP: // 鼠标右键松开
        if (prLabelState == IN_PROCESS) {
            setLabelInMask(flags, cv::Point(x, y), true);
            prLabelState = SET;
            showImage();
        }
        break;
    case cv::EVENT_MOUSEMOVE: // 鼠标移动
        if (rectState == IN_PROCESS) {
            rect = cv::Rect(cv::Point(rect.x, rect.y), cv::Point(x, y));
            CV_Assert(bgdPxls.empty() && fgdPxls.empty() && prBgdPxls.empty() && prFgdPxls.empty());
            showImage();
        } else if (labelState == IN_PROCESS) {
            setLabelInMask(flags, cv::Point(x, y), false);
            showImage();
        } else if (prLabelState == IN_PROCESS) {
            setLabelInMask(flags, cv::Point(x, y), true);
            showImage();
        }
        break;
    }
}
int UIApp::runGrabCut(int iterCount, lxh::GMMInitType initType, bool useRefine)
{
    
    if (isInitialized){
        if (useRefine){
            cout << "\033[32m" <<  "-------------- REFINE ------------" << "\033[30m" << endl;
            appTimer.start();
            grabCut->runGrabCut(mask, bgdModel, fgdModel, lxh::GC_REFINE, initType);
            appTimer.end("Performing REFINE iterations", true);
            nowIterCount += 1;
        }
        else{
            cout << "\033[32m" <<  "-------------- EDIT ------------" << "\033[30m" << endl;
            appTimer.start();
            grabCut->runGrabCut(mask, bgdModel, fgdModel, lxh::GC_EDIT, initType, 0, &bgdPxls, &fgdPxls);
            appTimer.end("Performing EDIT iterations", true);
            nowIterCount += 1;
        }
    }
    else {
        if (rectState != SET)
            return nowIterCount;
        // 不需要传Rect，Rect已经使用setRectInMask嵌入Mask了
        cout << "\033[32m" << "-------------- INIT_GrabCut_WITH_MASK ------------" << "\033[30m" << endl;
        appTimer.start();
        grabCut->runGrabCut(mask, bgdModel, fgdModel, lxh::GC_INIT, initType, iterCount);
        appTimer.end("Performing GrabCut with " + to_string(iterCount) + " INIT iterations", true);
        isInitialized = true;
        nowIterCount += iterCount;
    }
    
    
    bgdPxls.clear();
    fgdPxls.clear();
    prBgdPxls.clear();
    prFgdPxls.clear();
    return nowIterCount;
}
int UIApp::runGraphCut()
{
    if (rectState != SET)
        return nowIterCount;
    // 不需要传Rect，Rect已经使用setRectInMask嵌入Mask了
    cout << "\033[32m" << "-------------- INIT_GraphCut_WITH_MASK ------------" << "\033[30m" << endl;
    appTimer.start();
    grabCut->runGraphCut(mask);
    appTimer.end("Performing GraphCut", true);

    isInitialized = true;
    bgdPxls.clear();
    fgdPxls.clear();
    prBgdPxls.clear();
    prFgdPxls.clear();
    return nowIterCount+1;
}


