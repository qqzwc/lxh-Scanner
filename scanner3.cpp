#include "lxh_app.hpp"
#include <stdio.h>
using namespace std;

const char IMG_PATH[] = "test_imgs/test3.jpg";

const int ITER_COUNT = 3; // 单次迭代次数
const bool USE_REFINE = true;
const double gamma = 50;
UIApp myApp(gamma);

static void on_mouse(int event, int x, int y, int flags, void* param)
{
    myApp.mouseClick(event, x, y, flags, param);
}

int main(int argc, char** argv)
{
    cv::Mat image = cv::imread(IMG_PATH, cv::IMREAD_COLOR);
    if (image.empty()) {
        cout << "\033[31m"
             << "Couldn't read image from " << IMG_PATH << "!"
             << "\033[30m" << endl;
        return 1;
    }
    cv::resize(image, image, cv::Size(0, 0), 0.5, 0.5);
    const string winName = "FG(red,shift), BG(blue,ctrl)";
    cv::namedWindow(winName, cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback(winName, on_mouse, 0);
    myApp.setImageAndWinName(image, winName);
    myApp.showImage();
    bool ok = true;
    while (ok) {
        char c = (char)cv::waitKey(0);
        switch (c) {
        case '\x1b': // ESC退出
            cout << "\033[31m"
                 << "Finishing GrabCut ..."
                 << "\033[30m" << endl;
            ok = false;
            break;
        case 'r': // 重置
            cout << endl;
            myApp.reset();
            myApp.showImage();
            break;
        case ' ': // 按空格运行grabcut
            int oldIterCount = myApp.getIterCount();
            int newIterCount = myApp.runGrabCut(ITER_COUNT, lxh::KMEANS_RANDOM_INIT, USE_REFINE);

            if (newIterCount > oldIterCount) {
                myApp.showImage();
                cout << "\033[31m" << newIterCount << " iterations have been performed!"
                     << "\033[30m" << endl;
            } else
                cout << "\033[31m"
                     << "Rect must be determined!"
                     << "\033[30m" << endl;
            break;
        }
    }

    cv::Mat mask = myApp.getBinMask();
    for (int row = 0; row < mask.rows; ++row) {
        for (int col = 0; col < mask.cols; ++col) {
            if (mask.at<uchar>(row, col))
                mask.at<uchar>(row, col) = 255;
        }
    }
    // cv::imshow("Bin mask", mask);

    // 泛洪填充获得缺失像素区域
    cv::Mat tmpMask = cv::Mat::zeros(mask.rows + 2, mask.cols + 2, mask.type());
    mask.copyTo(tmpMask(cv::Range(1, mask.rows + 1), cv::Range(1, mask.cols + 1)));
    cv::floodFill(tmpMask, cv::Point(0, 0), cv::Scalar(125)); // 填成灰色以便于分辨（之前不是黑就是白）
    // cv::imshow("Tmp mask", tmpMask);
    tmpMask(cv::Range(1, mask.rows + 1), cv::Range(1, mask.cols + 1)).copyTo(mask);
    mask = ~(mask != 125); // 白色部分(255)为缺失像素点
    // 作膨胀腐蚀操作,加了以后能去掉一些噪点
    cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::Mat dilate_out;
    cv::dilate(mask, dilate_out, element);
    cv::dilate(dilate_out, dilate_out, element);
    cv::dilate(dilate_out, dilate_out, element);
    erode(dilate_out, mask, element);


    cv::imwrite("scanner3Mask.png", mask);

    return 0;
}
