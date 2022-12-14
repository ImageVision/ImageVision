#ifndef EZSIFT_H
#define EZSIFT_H

#include "image9.h"
#include <list>
#include <iostream>
#include <vector>
using namespace std;

namespace ezsift {

/****************************************
 * 常量参数
 ***************************************/
//每个Octave中包含的尺度s: 实际计算使用3-5中的值
static int SIFT_INTVLS = 3;

//高斯模糊Sigma
static float SIFT_SIGMA = 1.6f;

// 高斯核半径参数 (2*radius+1)x(2*radius+1), 通常使用2 或 3
static float SIFT_GAUSSIAN_FILTER_RADIUS = 3.0f;

// 对比度阈值 |D(x)|
static float SIFT_CONTR_THR = 8.0f; // 8.0f;

//主曲率阈值 r
static float SIFT_CURV_THR = 10.0f;

//子像素下阈值
static float SIFT_KEYPOINT_SUBPiXEL_THR = 0.6f;

// 金字塔构建前图像大小加倍
static bool SIFT_IMG_DBL = false; // true;

// 输入图像的假定高斯模糊
static float SIFT_INIT_SIGMA = 0.5f;

//边界阈值
static int SIFT_IMG_BORDER = 5;

// 最大插值步长
static int SIFT_MAX_INTERP_STEPS = 5;

// 方向分配直方图中的默认箱数
static int SIFT_ORI_HIST_BINS = 36;

// 确定方向分配的高斯西格玛
static float SIFT_ORI_SIG_FCTR =1.5f; //可能会影响方向计算 

// 确定方向指定中使用的区域的半径
static float SIFT_ORI_RADIUS =3 * SIFT_ORI_SIG_FCTR; //可能影响方向计算

// 相对于产生新特征的最大值的方向幅值
static float SIFT_ORI_PEAK_RATIO = 0.8f;

// 确定单个描述符方向直方图的大小
static float SIFT_DESCR_SCL_FCTR = 3.f;

// 描述向量元素大小阈值
static float SIFT_DESCR_MAG_THR = 0.2f;

//用于将浮点描述符转换为无符号字符的因子
static float SIFT_INT_DESCR_FCTR = 512.f;

// 关键点邻域子区域划分数
static int SIFT_DESCR_WIDTH = 4;

// 每个子区域的梯度划分方向
static int SIFT_DESCR_HIST_BINS = 8;

/****************** SIFT匹配参数 ********************/

// 用于匹配的近邻距离值阈值, 阈值越小匹配越精确, 0.1无匹配, |DR_nearest|/|DR_2nd_nearest|<SIFT_MATCH_NNDR_THR
static float SIFT_MATCH_NNDR_THR = 0.65f;

#if 0
//用于DoG金字塔的中间类型
typedef short sift_wt;
static const int SIFT_FIXPT_SCALE = 48;
#else
// 用于DoG金字塔的中间类型
typedef float sift_wt;
static const int SIFT_FIXPT_SCALE = 1;
#endif

/****************************************
 * 定义
 ***************************************/
#define DEGREE_OF_DESCRIPTORS (128)   //SIFT关键点: 128维
struct SiftKeypoint {
    int octave;   // octave数量
    int layer;    // layer数量
    float rlayer; // layer实际数量

    float r;     // 归一化的row坐标
    float c;     // 归一化的col坐标
    float scale; // 归一化的scale

    float ri;          // row坐标(layer)
    float ci;          // column坐标(layer)
    float layer_scale; // scale(layer)

    float ori; // 方向(degrees)
    float mag; // 模值

    float descriptors[DEGREE_OF_DESCRIPTORS]; //描述符 
};

struct MatchPair {
    int r1;
    int c1;
    int r2;
    int c2;
};

/****************************************
 *  SIFT处理函数 
 ***************************************/
void init_sift_parameters(bool doubleFirstOctave = true,
                          float contrast_threshold = 8.0f,
                          float edge_threshold = 10.0f,
                          float match_NDDR_threshold = 0.6f);

void double_original_image(bool doubleFirstOctave);

int gaussian_blur(const Image<float> &in_image, Image<float> &out_image,
                  std::vector<float> coef1d);

int row_filter_transpose(float *src, float *dst, int w, int h, float *coef1d,
                         int gR);

int build_octaves(const Image<unsigned char> &image,
                  std::vector<Image<unsigned char>> &octaves, int firstOctave,
                  int nOctaves);

std::vector<std::vector<float>> compute_gaussian_coefs(int nOctaves,
                                                       int nGpyrLayers);

int build_gaussian_pyramid(std::vector<Image<unsigned char>> &octaves,
                           std::vector<Image<float>> &gpyr, int nOctaves,
                           int nGpyrLayers);

int build_dog_pyr(std::vector<Image<float>> &gpyr,
                  std::vector<Image<float>> &dogPyr, int nOctaves,
                  int nDogLayers);

int build_grd_rot_pyr(std::vector<Image<float>> &gpyr,
                      std::vector<Image<float>> &grdPyr,
                      std::vector<Image<float>> &rotPyr, int nOctaves,
                      int nLayers,int USE_FAST_FUNC);

bool refine_local_extrema(std::vector<Image<float>> &dogPyr, int nOctaves,
                          int nDogLayers, SiftKeypoint &kpt);

float compute_orientation_hist(const Image<float> &image, SiftKeypoint &kpt,
                               float *&hist);

/****************************************
 *  SIFT核心函数 
 ***************************************/
int detect_keypoints(std::vector<Image<float>> &dogPyr,
                     std::vector<Image<float>> &grdPyr,
                     std::vector<Image<float>> &rotPyr, int nOctaves,
                     int nDogLayers, std::list<SiftKeypoint> &kpt_list);

int extract_descriptor(std::vector<Image<float>> &grdPyr,
                       std::vector<Image<float>> &rotPyr, int nOctaves,
                       int nGpyrLayers, std::list<SiftKeypoint> &kpt_list);

/****************************************
 *  SIFT接口函数 
 ***************************************/
int sift_cpu(const Image<unsigned char> &image,
             std::list<SiftKeypoint> &kpt_list, bool bExtractDescriptors,int USE_FAST_FUNC);

vector<int> match_keypoints(std::list<SiftKeypoint> &kpt_list1,
                    std::list<SiftKeypoint> &kpt_list2,
                    std::list<MatchPair> &match_list,int OptimizationSwitch);

}

#endif
