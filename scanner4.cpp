#include "lxh_rectangling.hpp"
#include <GL/glut.h> // GLUT头文件
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "lxh_app.hpp"

// 确保 x 在 [a, b]区间内
#define clamp(x, a, b) (((a) < (b))                     \
        ? ((x) < (a)) ? (a) : (((x) > (b)) ? (b) : (x)) \
        : ((x) < (b)) ? (b) : (((x) > (a)) ? (a) : (x)))


const int ITER_COUNT1 = 3; // GrabCut单次迭代次数
const bool USE_REFINE = true;
const double gamma = 50;
UIApp myApp(gamma);

const char IMG_PATH[] = "./test_imgs/test8.jpg";
const int ITER_COUNT2 = 10;
double lambdaL = 100, lambdaB = INF;
int rowMeshNum = 20, colMeshNum = 20; // 网格size

lxh::Timer timer; // 计时器
cv::Mat img; // 输入图像
vector<vector<lxh::CoordDouble>> outputMesh;
vector<vector<lxh::CoordDouble>> mesh;
double sx_avg = 1, sy_avg = 1;
GLuint texGround;

static void on_mouse(int event, int x, int y, int flags, void* param)
{
    myApp.mouseClick(event, x, y, flags, param);
}

GLuint matToTexture(cv::Mat mat, GLenum minFilter = GL_LINEAR,
    GLenum magFilter = GL_LINEAR, GLenum wrapFilter = GL_REPEAT)
{
    // cv::flip(mat, mat, 0);
    //  Generate a number for our textureID's unique handle
    GLuint textureID;
    glGenTextures(1, &textureID);

    // Bind to our texture handle
    glBindTexture(GL_TEXTURE_2D, textureID);

    // Catch silly-mistake texture interpolation method for magnification
    if (magFilter == GL_LINEAR_MIPMAP_LINEAR || magFilter == GL_LINEAR_MIPMAP_NEAREST || magFilter == GL_NEAREST_MIPMAP_LINEAR || magFilter == GL_NEAREST_MIPMAP_NEAREST) {
        // cout << "You can't use MIPMAPs for magnification - setting filter to GL_LINEAR" << endl;
        magFilter = GL_LINEAR;
    }

    // Set texture interpolation methods for minification and magnification
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, minFilter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, magFilter);

    // Set texture clamping method
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrapFilter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrapFilter);

    // Set incoming texture format to:
    // GL_BGR       for CV_CAP_OPENNI_BGR_IMAGE,
    // GL_LUMINANCE for CV_CAP_OPENNI_DISPARITY_MAP,
    // Work out other mappings as required ( there's a list in comments in main() )
    GLenum inputColourFormat = GL_BGR_EXT;
    if (mat.channels() == 1) {
        inputColourFormat = GL_LUMINANCE;
    }

    // Create the texture
    glTexImage2D(GL_TEXTURE_2D, // Type of texture
        0, // Pyramid level (for mip-mapping) - 0 is the top level
        GL_RGB, // Internal colour format to convert to
        mat.cols, // Image width  i.e. 640 for Kinect in standard mode
        mat.rows, // Image height i.e. 480 for Kinect in standard mode
        0, // Border width in pixels (can either be 1 or 0)
        inputColourFormat, // Input image format (i.e. GL_RGB, GL_RGBA, GL_BGR etc.)
        GL_UNSIGNED_BYTE, // Image data type
        mat.ptr()); // The actual image data itself

    // If we're using mipmaps then generate them. Note: This requires OpenGL 3.0 or higher

    return textureID;
}

void display(void)
{
    glLoadIdentity();
    // 清除屏幕
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glBindTexture(GL_TEXTURE_2D, texGround);
    for (int row = 0; row < rowMeshNum; row++) {
        for (int col = 0; col < colMeshNum; col++) {
            lxh::CoordDouble& localCoord = mesh[row][col];
            // 将local坐标标准化到[0,1]
            localCoord.row /= img.rows;
            localCoord.col /= img.cols;
            localCoord.row = clamp(localCoord.row, 0, 1);
            localCoord.col = clamp(localCoord.col, 0, 1);
            lxh::CoordDouble& globalCoord = outputMesh[row][col];
            // 将global坐标标准化到[-1,1]
            globalCoord.row /= img.rows;
            globalCoord.col /= img.cols;
            globalCoord.row -= 0.5;
            globalCoord.col -= 0.5;
            globalCoord.row *= 2;
            globalCoord.col *= 2;
            globalCoord.row = clamp(globalCoord.row, -1, 1);
            globalCoord.col = clamp(globalCoord.col, -1, 1);
        }
    }

    for (int row = 0; row < rowMeshNum - 1; row++) {
        for (int col = 0; col < colMeshNum - 1; col++) {
            lxh::CoordDouble local_left_top = mesh[row][col];
            lxh::CoordDouble local_right_top = mesh[row][col + 1];
            lxh::CoordDouble local_left_bottom = mesh[row + 1][col];
            lxh::CoordDouble local_right_bottom = mesh[row + 1][col + 1];

            lxh::CoordDouble global_left_top = outputMesh[row][col];
            lxh::CoordDouble global_right_top = outputMesh[row][col + 1];
            lxh::CoordDouble global_left_bottom = outputMesh[row + 1][col];
            lxh::CoordDouble global_right_bottom = outputMesh[row + 1][col + 1];

            glBegin(GL_QUADS);
            glTexCoord2d(local_right_top.col, local_right_top.row);
            glVertex3d(global_right_top.col, -1 * global_right_top.row, 0.0f);
            glTexCoord2d(local_right_bottom.col, local_right_bottom.row);
            glVertex3d(global_right_bottom.col, -1 * global_right_bottom.row, 0.0f);
            glTexCoord2d(local_left_bottom.col, local_left_bottom.row);
            glVertex3d(global_left_bottom.col, -1 * global_left_bottom.row, 0.0f);
            glTexCoord2d(local_left_top.col, local_left_top.row);
            glVertex3d(global_left_top.col, -1 * global_left_top.row, 0.0f);
            glEnd();
        }
    }
    glutSwapBuffers();
}

int main(int argc, char* argv[])
{
    #ifdef LXH_PARALLEL
    omp_set_num_threads(8); // 开启多线程加速
    #endif

    img = cv::imread(IMG_PATH, cv::IMREAD_COLOR);
    if (img.empty()) {
        cout << "\033[31m"
             << "Couldn't read image from " << IMG_PATH << "!"
             << "\033[30m" << endl;
        return 1;
    }
    cv::resize(img, img, cv::Size(0, 0), 0.2, 0.2);

    if (img.cols % 4 != 0)
        cv::resize(img, img, cv::Size(img.cols - (img.cols % 4), img.rows), 1, 1); // openGL需要图片宽度为4
    const string winName = "FG(red,shift), BG(blue,ctrl)";
    cv::namedWindow(winName, cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback(winName, on_mouse, 0);
    myApp.setImageAndWinName(img, winName);
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
            int newIterCount;
            newIterCount = myApp.runGrabCut(ITER_COUNT1, lxh::KMEANS_RANDOM_INIT, USE_REFINE);
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
    cv::Rect rect = myApp.getRect();
    cv::Mat mask = myApp.getBinMask()(rect);
    img = img(rect); // 只取矩形区域

    if (img.cols % 4 != 0){
        cv::resize(img, img, cv::Size(img.cols - (img.cols % 4), img.rows), 1, 1); // openGL需要图片宽度为4
        cv::resize(mask, mask, cv::Size(mask.cols - (mask.cols % 4), mask.rows), 1, 1);
    }
    cv::imshow("Origin Image", img);
    cv::waitKey(0);
	
    // double IMG_SCALE_FACTOR = sqrt(1000000.0 / (img.rows * img.cols)); 
    double IMG_SCALE_FACTOR = 1.0;
    cout << "IMG_SCALE_FACTOR: " << IMG_SCALE_FACTOR << endl;
    lxh::Rectangling rectangling(img, mask, IMG_SCALE_FACTOR, lambdaL, lambdaB, rowMeshNum, colMeshNum);
    timer.start();
    auto sx_sy_avg = rectangling.runRectangling(ITER_COUNT2, false);
    timer.end("Run rectangling", true);

    mesh = rectangling.getLocalMesh();
    outputMesh = rectangling.getGlobalMesh();

    cout << "img size " <<  img.size() << endl;
    

    // 使用OpenGL渲染
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(img.cols, img.rows); // 太大了显示不下
    glutInitWindowPosition(0, 0);
    glutCreateWindow("lxh-Rectangling");
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_TEXTURE_2D); // 启用纹理

    cv::waitKey(0);


    texGround = matToTexture(img);
    glutDisplayFunc(&display); //注册函数
    glutMainLoop();

    return 0;
}
