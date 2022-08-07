#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <algorithm>
#include <queue>
using namespace std;

namespace lxh{

const float FLOAT_EPS = 1e-30;
class ColorBin{
public:
    int x, y, z;
    float freq; // 频数
    ColorBin(int _x, int _y, int _z, float _freq){
        x = _x;
        y = _y;
        z = _z;
        freq = _freq;
    }
    bool operator<(const ColorBin& cb) const{
        return freq < cb.freq;
    }
};

class ColorHist{
private:
    static const int bin = 12; // 量化为12个值
    float allFreq;
    cv::Mat hist;
    float colorHist[bin][bin][bin];// 获取像素值对应的频数，占比不到百分之5的像素点映射到最近的占比超过百分之5的像素点的频数
    bool visited[bin][bin][bin];
    /*
    使用bfs寻找距离最近的有频数的点
    */
    float findFriend(int x, int y, int z){
        
        for(int xi=0; xi < bin; ++xi){
            for(int yi=0; yi < bin; ++yi){
                for(int zi=0; zi < bin; ++zi){
                    visited[xi][yi][zi] = false;
                }
            }
        }

        int direction[3] = {-1, 0, 1};
        queue<ColorBin> colorBinQueue;
        colorBinQueue.push(ColorBin(x, y, z, 0));
        visited[x][y][z] = true;

        int cnt = 0;
        while(!colorBinQueue.empty()){
            ColorBin tmpBin = colorBinQueue.front();
            colorBinQueue.pop();
            ++cnt;
            if(tmpBin.freq > FLOAT_EPS){
                return tmpBin.freq;
            }
            for(int xi=0; xi < 3; ++xi){
                for(int yi=0; yi < 3; ++yi){
                    for(int zi=0; zi < 3; ++zi){
                        if (!(xi == 0 && yi == 0 && zi == 0)){
                            int newx = tmpBin.x + direction[xi];
                            if (newx < 0 || newx >= bin)
                                continue;
                            int newy = tmpBin.y + direction[yi];
                            if (newy < 0 || newy >= bin)
                                continue;
                            int newz = tmpBin.z + direction[zi];
                            if (newz < 0 || newz >= bin)
                                continue;
 
                            if (visited[newx][newy][newz])
                                continue;

                            colorBinQueue.push(ColorBin(newx, newy, newz, colorHist[newx][newy][newz]));
                            visited[newx][newy][newz] = true;
                        }
                    }
                }
            }
        }
        cout << cnt << endl;
        CV_Error(cv::Error::BadCallBack, "Shouldn't execute this!");
        return FLOAT_EPS;
    }
public:
    ColorHist(const cv::Mat& img, const cv::Mat& mask, float keep=0.95){
        int histSize[] = { bin, bin, bin };
        float Branges[] = { 0, 255 };
        float Granges[] = { 0, 255 };
        float Rranges[] = { 0, 255 };
        const float* ranges[] = { Branges, Granges, Rranges };
        int channels[] = { 0, 1, 2 };
        cv::calcHist(&img, 1, channels, mask, hist, 3, histSize, ranges, true, false);
        vector<ColorBin> colorBins;
        allFreq = 0;
        for(int x = 0; x < bin; ++x){
            for(int y = 0; y < bin; ++y){
                for(int z = 0; z < bin; ++z){
                    float tmpFreq = hist.at<float>(x, y, z);
                    if(tmpFreq > FLOAT_EPS){
                        colorBins.push_back(ColorBin(x, y, z, tmpFreq));
                        allFreq += tmpFreq;
                    }
                    colorHist[x][y][z] = 0; // 初始化
                }
            }
        }
        sort(colorBins.begin(), colorBins.end());
        float targetFreq = keep * allFreq;
        int i = 0;
        while((allFreq - colorBins[i].freq) > targetFreq){
            allFreq -= colorBins[i].freq;
            ++i;
        }
        cout << "The number of Color Hist: bins: " << int(colorBins.size()) - i << endl;
        while(i < colorBins.size()){
            ColorBin tmpBin = colorBins[i];
            colorHist[tmpBin.x][tmpBin.y][tmpBin.z] = tmpBin.freq;
            ++i;
        }
        
        // 解决没有分配的点的归属
        float debbb = 0;
        for(int x = 0; x < bin; ++x){
            for(int y = 0; y < bin; ++y){
                for(int z = 0; z < bin; ++z){
                    if(colorHist[x][y][z] < FLOAT_EPS){
                        // cout << x << " " << y << " " << z  << endl;
                        colorHist[x][y][z] = findFriend(x, y, z);
                        
                    }
                    else
                        debbb += colorHist[x][y][z];
                }
            }
        }
    }
    float prob(const cv::Vec3b _color) const{
        int x = int(bin * _color[0] / 256);
        int y = int(bin * _color[1] / 256);
        int z = int(bin * _color[2] / 256);
        return colorHist[x][y][z]/ allFreq;
    }
};


}

// const char IMG_PATH[] = "./testfile/cat_600x400.png";
// int main()
// {
//     cv::Mat img = cv::imread(IMG_PATH, cv::IMREAD_COLOR);
//     ColorHist a(img);
//     for(int i = 0; i < 100; ++i)
//     cout << a.prob(img.at<cv::Vec3b>(i, i)) << endl;

//     return 0;
// }