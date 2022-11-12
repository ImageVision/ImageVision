#include "CImg.h"
#include <iostream>
using namespace std;
using namespace cimg_library;
#include "ext.h"

//多种格式互转
int main() 
{
	CImg<unsigned char> img;
	img = readImage("input.jpg");  //支持多种格式，input.bmp、input.png、input.jpg、input.tga 
	saveImage("output.bmp", img);   //支持多种格式，output.bmp、output.png、output.jpg、output.tga
	return 0;
}
