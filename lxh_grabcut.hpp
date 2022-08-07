/*
GrabCut算法实现
参考opencv源码：https://github.com/opencv/opencv/blob/e628fd7bce2b5d64c36b5bdc55a37c0ae78bc907/modules/imgproc/src/grabcut.cpp
*/
#pragma once
#include "lxh_GMM.hpp"
#include "lxh_colorhist.hpp"
#include "lxh_timer.hpp"
#include "maxflow/block.h"
#include "maxflow/graph.cpp"
#include "maxflow/graph.h"

namespace lxh {

#ifdef LXH_OUTPUT
fstream fGCLog("./output/lxh_grabCut.log", ios::out);
#endif

lxh::Timer gTimer(true);

typedef maxflow::Graph<double, double, double> GCGraph;
typedef maxflow::Block<int> GCBlock;

enum PixelType {
    BGD, // 背景，矩形框外的区域以及后面用背景刷刷的区域
    FGD, // 前景，后面用前景刷刷的区域
    PR_BGD, // 可能的背景，不人为设定的话是用不上的
    PR_FGD // 可能的前景，矩形框内的区域
};

enum GMMInitType {
    KMEANS_PP_INIT,
    KMEANS_RANDOM_INIT
    /*
    原始k-means算法最开始随机选取数据集中k个点作为聚类中心，而k-means++按照如下的思想选取K个聚类中心：
    假设已经选取了n个初始聚类中心(0<n<k)，则在选取第n+1个聚类中心时：距离当前n个聚类中心越远的点会有更高的概率被选为第n+1个聚类中心。
    在选取第一个聚类中心(n=1)时同样通过随机的方法。
    可以说这也符合我们的直觉：聚类中心当然是互相离得越远越好。这个改进虽然直观简单，但是却非常得有效。
    */
};

enum GCMode {
    GC_INIT, // 从头开始运行GrabCut算法
    GC_EDIT, // 用户编辑后更新t-links后再执行一次min-cut
    GC_REFINE // 用户编辑后更新t-links后再执行一次完整的迭代
};

class GrabCut2D {

private:
    cv::Mat img; // 图像
    cv::Mat leftW, upleftW, upW, uprightW; // n-links 权值
    double gamma;
    double K;
    double beta;
    GCGraph* gcGraph;
    GCBlock* changedList; // TODO 以后考虑实现
    int recorderCount;

public:
    GrabCut2D(cv::InputArray _img, double _gamma = 50)
    {
        img = _img.getMat();
        if (img.empty())
            CV_Error(cv::Error::StsBadArg, "image is empty");
        if (img.type() != CV_8UC3)
            CV_Error(cv::Error::StsBadArg, "image must have CV_8UC3 type");
        gcGraph = nullptr; // 未建图设为nullptr
        gamma = _gamma;
        K = 9 * gamma; // 一个像素点最多连8条边，每条边的权值最大是gamma，因而K设为9 * gamma就比所有的n-links权值都大
        calcBeta();
        gTimer.start();
        calcNlinksWeights();
        gTimer.end("Calculate n-links weights");
        recorderCount = 0;
        changedList = new GCBlock(img.rows * img.cols); // 考虑最大需要情况
    }

    ~GrabCut2D()
    {
        if (gcGraph != nullptr)
            delete gcGraph;
        if (changedList != nullptr)
            delete changedList;
    }

    void runGrabCut(cv::InputOutputArray _mask,
        cv::InputOutputArray _bgdModel, cv::InputOutputArray _fgdModel,
        GCMode mode = GC_INIT, GMMInitType initType = KMEANS_PP_INIT, int iterCount = 1,
        vector<cv::Point>* _bgdPxls = nullptr, vector<cv::Point>* _fgdPxls = nullptr)
    {
        cv::Mat& mask = _mask.getMatRef();
        checkMask(mask);
        cv::Mat& bgdModel = _bgdModel.getMatRef();
        cv::Mat& fgdModel = _fgdModel.getMatRef();
        GMM3D bgdGMM(bgdModel), fgdGMM(fgdModel);

        if (mode == GC_INIT) {
#ifdef LXH_OUTPUT
            fGCLog << "-------------- GC_INIT --------------" << endl;
#endif
            CV_Assert(iterCount > 0);
            gTimer.start();
            initGMMs(initType, mask, bgdGMM, fgdGMM);
            gTimer.end("Init GMM");

            for (int i = 0; i < iterCount; i++) {
                cv::Mat clusterLabels(img.size(), CV_32SC1);
                gTimer.start();
                assignGMMsComponents(mask, bgdGMM, fgdGMM, clusterLabels);
#ifdef LXH_OUTPUT
                cv::Mat showLabels;
                clusterLabels.copyTo(showLabels);
                cv::Point p;
                for (p.y = 0; p.y < img.rows; p.y++) {
                    for (p.x = 0; p.x < img.cols; p.x++) {
                        cv::Vec3d color = img.at<cv::Vec3b>(p);
                        if (mask.at<uchar>(p) == BGD || mask.at<uchar>(p) == PR_BGD)
                            showLabels.at<int>(p) = clusterLabels.at<int>(p) + 5; // 为了展示时区分前后景加个5
                        else
                            showLabels.at<int>(p) = clusterLabels.at<int>(p);
                    }
                }
                cv::imwrite("./output/clusterLabels/GC_INIT_labels" + to_string(++recorderCount) + ".png", showLabels);
#endif
                gTimer.end("Assign GMMs Components weights");

                gTimer.start();
                learnGMMs(mask, clusterLabels, bgdGMM, fgdGMM); // TODO输出GMM的标签变化过程
                gTimer.end("Learn GMMs");

                gTimer.start();
                constructGCGraph(mask, bgdGMM, fgdGMM);
                gTimer.end("Construct GCGraph");

                gTimer.start();
                calcMaxflow();
                gTimer.end("Calculate Maxflow");

                gTimer.start();
                estimateSegmentation(mask);
                gTimer.end("Estimate Segmentation");
            }
        } else if (mode == GC_EDIT) { // 除了从头开始INIT其他模式都只需要迭代一次
// 修改图结构后再求一次分割
#ifdef LXH_OUTPUT
            fGCLog << "-------------- GC_EDIT --------------" << endl;
#endif
            gTimer.start();
            constructGCGraph(mask, bgdGMM, fgdGMM, true, _bgdPxls, _fgdPxls); // TODO 修改图有问题
            gTimer.end("Modify GCGraph");

            gTimer.start();
            calcMaxflow(true);
            gTimer.end("Calculate Maxflow");

            gTimer.start();
            estimateSegmentation(mask, true); // 其实预测这里还有提升空间
            gTimer.end("Estimate Segmentation");
        } else if (mode == GC_REFINE) { // Refine就直接再迭代一次
            cv::Mat clusterLabels(img.size(), CV_32SC1);
            gTimer.start();
            assignGMMsComponents(mask, bgdGMM, fgdGMM, clusterLabels);
            gTimer.end("Assign GMMs Components weights");
#ifdef LXH_OUTPUT
            fGCLog << "-------------- GC_REFINE --------------" << endl;
            cv::Mat showLabels;
            clusterLabels.copyTo(showLabels);
            cv::Point p;
            for (p.y = 0; p.y < img.rows; p.y++) {
                for (p.x = 0; p.x < img.cols; p.x++) {
                    cv::Vec3d color = img.at<cv::Vec3b>(p);
                    if (mask.at<uchar>(p) == BGD || mask.at<uchar>(p) == PR_BGD)
                        showLabels.at<int>(p) = clusterLabels.at<int>(p) + 5; // 为了展示时区分前后景加个5
                    else
                        showLabels.at<int>(p) = clusterLabels.at<int>(p);
                }
            }
            cv::imwrite("./output/clusterLabels/GC_REFINE_labels" + to_string(recorderCount) + ".png", showLabels);
#endif
            gTimer.start();
            learnGMMs(mask, clusterLabels, bgdGMM, fgdGMM); // TODO输出GMM的标签变化过程
            gTimer.end("Learn GMMs");

            gTimer.start();
            constructGCGraph(mask, bgdGMM, fgdGMM);
            gTimer.end("Construct GCGraph");

            gTimer.start();
            calcMaxflow();
            gTimer.end("Calculate Maxflow");

            gTimer.start();
            estimateSegmentation(mask);
            gTimer.end("Estimate Segmentation");
        } else {
            CV_Error(cv::Error::StsBadArg, "Not support this GC Mode!");
        }
    }

    void runGraphCut(cv::InputOutputArray _mask)
    {
        cv::Mat& mask = _mask.getMatRef();
        checkMask(mask);
        cv::Mat fgdMask, bgdMask;
        gTimer.start();
        getFgdBgdMask(mask, fgdMask, bgdMask);
        ColorHist fgdColorHist(img, fgdMask), bgdColorHist(img, bgdMask);
        gTimer.end("Init ColorHist");

        gTimer.start();
        constructGCGraphWithColorHist(mask, fgdColorHist, bgdColorHist);
        gTimer.end("Construct GCGraph With Color Hist");

        gTimer.start();
        calcMaxflow();
        gTimer.end("Calculate Maxflow");

        gTimer.start();
        estimateSegmentation(mask);
        gTimer.end("Estimate Segmentation");
    }

private:
    /*
    对给定的mask计算前景背景掩码
    */
    void getFgdBgdMask(const cv::Mat& mask, cv::Mat& fgdMask, cv::Mat& bgdMask)
    {
        if (mask.empty() || mask.type() != CV_8UC1)
            CV_Error(cv::Error::StsBadArg, "mask is empty or has incorrect type (not CV_8UC1)");
        if (fgdMask.empty() || fgdMask.rows != mask.rows || fgdMask.cols != mask.cols)
            fgdMask.create(mask.size(), CV_8UC1);
        if (bgdMask.empty() || bgdMask.rows != mask.rows || bgdMask.cols != mask.cols)
            bgdMask.create(mask.size(), CV_8UC1);
        cv::Point p;
        for (p.y = 0; p.y < mask.rows; p.y++) {
            for (p.x = 0; p.x < mask.cols; p.x++) {
                if (mask.at<uchar>(p) == BGD || mask.at<uchar>(p) == PR_BGD) {
                    fgdMask.at<uchar>(p) = 0;
                    bgdMask.at<uchar>(p) = 1;
                } else {
                    fgdMask.at<uchar>(p) = 1;
                    bgdMask.at<uchar>(p) = 0;
                }
            }
        }
    }

    /*
    Calculate beta - parameter of GrabCut algorithm.
    beta = 1/(2*avg(sqr(||color[i] - color[j]||)))
    只需要算每个点 左，左上，上，右上 四个方向的边就可以，不会漏掉也没有重复计算。
    */
    void calcBeta()
    {
        beta = 0;
        for (int y = 0; y < img.rows; y++) {
            for (int x = 0; x < img.cols; x++) {
                cv::Vec3d color = img.at<cv::Vec3b>(y, x);
                if (x > 0) { // left
                    cv::Vec3d diff = color - (cv::Vec3d)img.at<cv::Vec3b>(y, x - 1);
                    beta += diff.dot(diff);
                }
                if (y > 0 && x > 0) { // upleft
                    cv::Vec3d diff = color - (cv::Vec3d)img.at<cv::Vec3b>(y - 1, x - 1);
                    beta += diff.dot(diff);
                }
                if (y > 0) { // up
                    cv::Vec3d diff = color - (cv::Vec3d)img.at<cv::Vec3b>(y - 1, x);
                    beta += diff.dot(diff);
                }
                if (y > 0 && x < img.cols - 1) { // upright
                    cv::Vec3d diff = color - (cv::Vec3d)img.at<cv::Vec3b>(y - 1, x + 1);
                    beta += diff.dot(diff);
                }
            }
        }
        if (beta <= std::numeric_limits<double>::epsilon())
            beta = 0;
        else // 4mn-3m-3n+2是图中边的数量，4mn是四个方向的边，后面是去掉了边界上不存在而被4mn算进去的边
            beta = 1.f / (2 * beta / (4 * img.cols * img.rows - 3 * img.cols - 3 * img.rows + 2));
    }

    /*
    Calculate weights of no-terminal vertices of graph.
    即计算n-links边的权重，也叫做边界项、平滑项或者交互项
    gammaDivSqrt2相当于公式（4）中的gamma * dist(i,j)^(-1)，那么可以知道，
    当i和j是垂直或者水平关系时，dist(i,j)=1；
    当是对角关系时，dist(i,j)=sqrt(2.0f)。
    */
    void calcNlinksWeights()
    {
        const double gammaDivSqrt2 = gamma / std::sqrt(2.0f);
        leftW.create(img.rows, img.cols, CV_64FC1);
        upleftW.create(img.rows, img.cols, CV_64FC1);
        upW.create(img.rows, img.cols, CV_64FC1);
        uprightW.create(img.rows, img.cols, CV_64FC1);
        for (int y = 0; y < img.rows; y++) {
            for (int x = 0; x < img.cols; x++) {
                cv::Vec3d color = img.at<cv::Vec3b>(y, x);
                if (x - 1 >= 0) { // left
                    cv::Vec3d diff = color - (cv::Vec3d)img.at<cv::Vec3b>(y, x - 1);
                    leftW.at<double>(y, x) = gamma * exp(-beta * diff.dot(diff));
                } else
                    leftW.at<double>(y, x) = 0;
                if (x - 1 >= 0 && y - 1 >= 0) { // upleft
                    cv::Vec3d diff = color - (cv::Vec3d)img.at<cv::Vec3b>(y - 1, x - 1);
                    upleftW.at<double>(y, x) = gammaDivSqrt2 * exp(-beta * diff.dot(diff));
                } else
                    upleftW.at<double>(y, x) = 0;
                if (y - 1 >= 0) { // up
                    cv::Vec3d diff = color - (cv::Vec3d)img.at<cv::Vec3b>(y - 1, x);
                    upW.at<double>(y, x) = gamma * exp(-beta * diff.dot(diff));
                } else
                    upW.at<double>(y, x) = 0;
                if (x + 1 < img.cols && y - 1 >= 0) { // upright
                    cv::Vec3d diff = color - (cv::Vec3d)img.at<cv::Vec3b>(y - 1, x + 1);
                    uprightW.at<double>(y, x) = gammaDivSqrt2 * exp(-beta * diff.dot(diff));
                } else
                    uprightW.at<double>(y, x) = 0;
            }
        }
    }

    /*
    Initialize GMM background and foreground models using kmeans algorithm or random init.
    opencv默认是使用kmeans初始化，这里还尝试使用随机初始化（主要是出于速度上的考虑）
    */
    void initGMMs(const GMMInitType initType, const cv::Mat& mask, GMM3D& bgdGMM, GMM3D& fgdGMM)
    {
        cv::Mat bgdLabels, fgdLabels;
        std::vector<cv::Vec3f> bgdSamples, fgdSamples;
        cv::Point p;
        for (p.y = 0; p.y < img.rows; p.y++) {
            for (p.x = 0; p.x < img.cols; p.x++) {
                if (mask.at<uchar>(p) == BGD || mask.at<uchar>(p) == PR_BGD)
                    bgdSamples.push_back((cv::Vec3f)img.at<cv::Vec3b>(p));
                else // FGD | PR_FGD
                    fgdSamples.push_back((cv::Vec3f)img.at<cv::Vec3b>(p));
            }
        }
        CV_Assert(!bgdSamples.empty() && !fgdSamples.empty());
        // 将样本vector转换成样本矩阵
        cv::Mat _bgdSamples((int)bgdSamples.size(), 3, CV_32FC1, &bgdSamples[0][0]);
        cv::Mat _fgdSamples((int)fgdSamples.size(), 3, CV_32FC1, &fgdSamples[0][0]);
        int KMeansInitType = 0;
        if (initType == KMEANS_PP_INIT) {
            KMeansInitType = cv::KMEANS_PP_CENTERS;
        } else if (initType == KMEANS_RANDOM_INIT) {
            KMeansInitType = cv::KMEANS_RANDOM_CENTERS;
        } else {
            CV_Error(cv::Error::StsBadArg, "Not support this initType!");
        }

        const int kMeansItCount = 10; // 默认迭代10次
        cv::kmeans(_bgdSamples, GMM3D::componentsCount, bgdLabels,
            cv::TermCriteria(cv::TermCriteria::MAX_ITER, kMeansItCount, 0.0), 0, KMeansInitType);
        cv::kmeans(_fgdSamples, GMM3D::componentsCount, fgdLabels,
            cv::TermCriteria(cv::TermCriteria::MAX_ITER, kMeansItCount, 0.0), 0, KMeansInitType);

        bgdGMM.initLearning();
        for (int i = 0; i < (int)bgdSamples.size(); i++)
            bgdGMM.addSample(bgdLabels.at<int>(i, 0), bgdSamples[i]);
        bgdGMM.endLearning();

        fgdGMM.initLearning();
        for (int i = 0; i < (int)fgdSamples.size(); i++)
            fgdGMM.addSample(fgdLabels.at<int>(i, 0), fgdSamples[i]);
        fgdGMM.endLearning();
    }

    /*
     Assign GMMs components for each pixel.
     给每一个像素打上标签，即执行EM算法中的E阶段，这里为简化计算直接硬分类了
     */
    void assignGMMsComponents(const cv::Mat& mask, const GMM3D& bgdGMM, const GMM3D& fgdGMM, cv::Mat& clusterLabels)
    {
        cv::Point p;
        for (p.y = 0; p.y < img.rows; p.y++) {
            for (p.x = 0; p.x < img.cols; p.x++) {
                cv::Vec3d color = img.at<cv::Vec3b>(p);
                clusterLabels.at<int>(p) = mask.at<uchar>(p) == BGD || mask.at<uchar>(p) == PR_BGD ? bgdGMM.whichComponent(color) : fgdGMM.whichComponent(color);
            }
        }
    }

    /*
    Learn GMMs parameters.
    即执行EM算法中的M阶段
    */
    void learnGMMs(const cv::Mat& mask, const cv::Mat& clusterLabels, GMM3D& bgdGMM, GMM3D& fgdGMM)
    {
        bgdGMM.initLearning();
        fgdGMM.initLearning();
        cv::Point p;
        for (int ci = 0; ci < GMM3D::componentsCount; ci++) {
            for (p.y = 0; p.y < img.rows; p.y++) {
                for (p.x = 0; p.x < img.cols; p.x++) {
                    if (clusterLabels.at<int>(p) == ci) {
                        if (mask.at<uchar>(p) == BGD || mask.at<uchar>(p) == PR_BGD)
                            bgdGMM.addSample(ci, img.at<cv::Vec3b>(p));
                        else
                            fgdGMM.addSample(ci, img.at<cv::Vec3b>(p));
                    }
                }
            }
        }
        bgdGMM.endLearning();
        fgdGMM.endLearning();
    }

    /*
    Construct GCGraph
    构造GrabCut算法所需的图结构, 是GrabCut算法中相对耗时的一个步骤。

    */
    void constructGCGraph(const cv::Mat& mask, const GMM3D& bgdGMM, const GMM3D& fgdGMM, bool modify = false,
        vector<cv::Point>* _bgdPxls = nullptr, vector<cv::Point>* _fgdPxls = nullptr)
    {
        cv::Point p;
        if (modify) {
            for (int i = 0; i < _bgdPxls->size(); ++i) { // 采用GraphCut中的方式
                p = (*_bgdPxls)[i];
                int nodeIdx = p.y * img.cols + p.x;
                cv::Vec3b color = img.at<cv::Vec3b>(p);
                gcGraph->add_tweights(nodeIdx, -log(fgdGMM.prob(color)), -log(bgdGMM.prob(color) + K));
                gcGraph->mark_node(nodeIdx);
            }

            for (int i = 0; i < _fgdPxls->size(); ++i) {
                p = (*_fgdPxls)[i];
                int nodeIdx = p.y * img.cols + p.x;
                cv::Vec3b color = img.at<cv::Vec3b>(p);
                gcGraph->add_tweights(nodeIdx, -log(fgdGMM.prob(color)) + K, -log(bgdGMM.prob(color)));
                gcGraph->mark_node(nodeIdx);
            }
        } else {
            int nodeCount = img.cols * img.rows,
                edgeCount = 2 * (4 * img.cols * img.rows - 3 * (img.cols + img.rows) + 2); // 无向图，使用两个单向边来实现 TODO 应该是这意思吧

            // 新建或重新建图
            if (gcGraph != nullptr)
                delete gcGraph;
            gcGraph = new GCGraph(nodeCount, edgeCount);
            for (p.y = 0; p.y < img.rows; p.y++) {
                for (p.x = 0; p.x < img.cols; p.x++) {
                    // add node
                    int nodeIdx = gcGraph->add_node();
                    cv::Vec3b color = img.at<cv::Vec3b>(p);

                    // set t-links weights
                    double fromSource, toSink;
                    if (mask.at<uchar>(p) == PR_BGD || mask.at<uchar>(p) == PR_FGD) {
                        fromSource = -log(bgdGMM.prob(color));
                        toSink = -log(fgdGMM.prob(color));
                    } else if (mask.at<uchar>(p) == BGD) {
                        fromSource = 0;
                        toSink = K;
                    } else // GC_FGD
                    {
                        fromSource = K;
                        toSink = 0;
                    }
                    gcGraph->add_tweights(nodeIdx, fromSource, toSink);

                    // set n-links weights
                    if (p.x > 0) {
                        double w = leftW.at<double>(p);
                        gcGraph->add_edge(nodeIdx, nodeIdx - 1, w, w);
                    }
                    if (p.x > 0 && p.y > 0) {
                        double w = upleftW.at<double>(p);
                        gcGraph->add_edge(nodeIdx, nodeIdx - img.cols - 1, w, w);
                    }
                    if (p.y > 0) {
                        double w = upW.at<double>(p);
                        gcGraph->add_edge(nodeIdx, nodeIdx - img.cols, w, w);
                    }
                    if (p.x < img.cols - 1 && p.y > 0) {
                        double w = uprightW.at<double>(p);
                        gcGraph->add_edge(nodeIdx, nodeIdx - img.cols + 1, w, w);
                    }
                }
            }
        }
    }

    /*
      Construct GCGraph With ColorHist
      构造GraphCut算法所需的图结构, 使用彩色直方图。
      */
    void constructGCGraphWithColorHist(const cv::Mat& mask, const ColorHist& fgdColorHist, const ColorHist& bgdColorHist)
    {
        int nodeCount = img.cols * img.rows,
            edgeCount = 2 * (4 * img.cols * img.rows - 3 * (img.cols + img.rows) + 2); // 无向图，使用两个单向边来实现 TODO 应该是这意思吧

        // 新建或重新建图
        if (gcGraph != nullptr)
            delete gcGraph;
        gcGraph = new GCGraph(nodeCount, edgeCount);
        cv::Point p;
        for (p.y = 0; p.y < img.rows; p.y++) {
            for (p.x = 0; p.x < img.cols; p.x++) {
                // add node
                int nodeIdx = gcGraph->add_node();
                cv::Vec3b color = img.at<cv::Vec3b>(p);

                // set t-links weights
                double fromSource, toSink;
                if (mask.at<uchar>(p) == PR_BGD || mask.at<uchar>(p) == PR_FGD) {
                    fromSource = -log(bgdColorHist.prob(color));
                    toSink = -log(fgdColorHist.prob(color));
                } else if (mask.at<uchar>(p) == BGD) {
                    fromSource = 0;
                    toSink = K;
                } else // GC_FGD
                {
                    fromSource = K;
                    toSink = 0;
                }
                gcGraph->add_tweights(nodeIdx, fromSource, toSink);

                // set n-links weights
                if (p.x > 0) {
                    double w = leftW.at<double>(p);
                    gcGraph->add_edge(nodeIdx, nodeIdx - 1, w, w);
                }
                if (p.x > 0 && p.y > 0) {
                    double w = upleftW.at<double>(p);
                    gcGraph->add_edge(nodeIdx, nodeIdx - img.cols - 1, w, w);
                }
                if (p.y > 0) {
                    double w = upW.at<double>(p);
                    gcGraph->add_edge(nodeIdx, nodeIdx - img.cols, w, w);
                }
                if (p.x < img.cols - 1 && p.y > 0) {
                    double w = uprightW.at<double>(p);
                    gcGraph->add_edge(nodeIdx, nodeIdx - img.cols + 1, w, w);
                }
            }
        }
    }
    /*
    使用"An experimental comparison of min-cut/max-flow algorithms for energy minimization in vision"
    提出的算法计算最大流，在后面的
    */
    void calcMaxflow(bool reuseGraph = false)
    {
        CV_Assert(gcGraph != nullptr);
        if (reuseGraph) {
            CV_Assert(changedList != nullptr);
            cout << "Energy: " << gcGraph->maxflow(true, changedList) << endl;
#ifdef LXH_OUTPUT
            fGCLog << "Energy: " << gcGraph->maxflow(true, changedList) << endl;
#endif
        } else {
            cout << "Energy: " << gcGraph->maxflow() << endl;
#ifdef LXH_OUTPUT
            fGCLog << "Energy: " << gcGraph->maxflow() << endl;
#endif
        }
    }

    /*
    根据s-t割的分割结果划分为PR_FGD, PR_BGD
    */
    void estimateSegmentation(cv::Mat& mask, bool modify = false)
    { 
        CV_Assert(gcGraph != nullptr);
        cv::Point p;
        if (modify) {
            int* ptr;
            for (ptr = changedList->ScanFirst(); ptr; ptr = changedList->ScanNext()) {
                int nodeIdx = *ptr;  // nodeIdx = p.y * img.cols + p.x;
                cout << nodeIdx << endl;
                p.x = nodeIdx % img.cols;
                p.y = nodeIdx / img.cols;
                gcGraph->remove_from_changed_list(nodeIdx);
                if (gcGraph->what_segment(nodeIdx)) {
                    mask.at<uchar>(p) = PR_BGD;
                } else {
                    mask.at<uchar>(p) = PR_FGD;
                }
            }
            changedList->Reset();
        } else {
            for (p.y = 0; p.y < mask.rows; p.y++) {
                for (p.x = 0; p.x < mask.cols; p.x++) {
                    if (mask.at<uchar>(p) == PR_BGD || mask.at<uchar>(p) == PR_FGD) { // 只修改不能确定的区域
                        if (gcGraph->what_segment(p.y * mask.cols + p.x /*vertex index*/))
                            mask.at<uchar>(p) = PR_BGD;
                        else
                            mask.at<uchar>(p) = PR_FGD;
                    }
                }
            }
        }
    }

    /*
    Check size, type and element values of mask matrix.
    mask只能取以下四种值：
    BGD（=0），背景；
    FGD（=1），前景；
    PR_BGD（=2），可能的背景；
    PR_FGD（=3），可能的前景。
    */
    void checkMask(const cv::Mat& mask)
    {
        if (mask.empty())
            CV_Error(cv::Error::StsBadArg, "mask is empty");
        if (mask.type() != CV_8UC1)
            CV_Error(cv::Error::StsBadArg, "mask must have CV_8UC1 type");
        if (mask.cols != img.cols || mask.rows != img.rows)
            CV_Error(cv::Error::StsBadArg, "mask must have as many rows and cols as img");
        for (int y = 0; y < mask.rows; y++) {
            for (int x = 0; x < mask.cols; x++) {
                uchar val = mask.at<uchar>(y, x);
                if (val != BGD && val != FGD && val != PR_BGD && val != PR_FGD)
                    CV_Error(cv::Error::StsBadArg, "mask element value must be equal "
                                                   "BGD or FGD or PR_BGD or PR_FGD");
            }
        }
    }
};

}