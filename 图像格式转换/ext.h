#ifndef _EXT_H_
#define _EXT_H_
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "CImg.h"
using namespace cimg_library;
#include <iostream>
using namespace std;

CImg<unsigned char> readImage(string imgPath);
void saveImage(string filename, CImg<unsigned char> cimg, int quality = 0);

CImg<unsigned char> readImage(string imgPath)
{
	int w, h, c;
	unsigned char *data = stbi_load(imgPath.c_str(), &w, &h, &c, 0);
	unsigned char* ndata = new unsigned char[w*h*c];
	int t = 0;
	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			for (int k = 0; k < c; k++)
			{
				ndata[w*h*k+t] = data[i*(w*c) + c*j + k];
			}
			t++;
		}
	}
	CImg<unsigned char> dst(ndata, w, h, 1, c);
	stbi_image_free(data);
	stbi_image_free(ndata);
	return dst;
}

void saveImage(string filename, CImg<unsigned char> cimg, int quality)
{
	int w = cimg._width;
	int h = cimg._height;
	int c = cimg._spectrum;
	std::string name(filename);


	unsigned char* ndata = cimg._data;
	unsigned char* data = new unsigned char[w*h*c];
	int t = 0;
	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			for (int k = 0; k < c; k++)
			{
				data[i*(w*c) + c*j + k] = ndata[w*h*k + t];
			}
			t++;
		}
	}
	std::string extension("jpg");
	int tp = name.compare(name.size() - extension.size(), extension.size(), extension) == 0;
	if (tp == 1)
	{
		stbi_write_jpg(filename.c_str(), w, h, c, data, quality);
		stbi_image_free(data);
		return;
	}
	extension = ("png");
	tp = name.compare(name.size() - extension.size(), extension.size(), extension) == 0;
	if (tp == 1)
	{
		stbi_write_png(filename.c_str(), w, h, c, data, quality);
		stbi_image_free(data);
		return;
	}
	extension = ("bmp");
	tp = name.compare(name.size() - extension.size(), extension.size(), extension) == 0;
	if (tp == 1)
	{
		stbi_write_bmp(filename.c_str(), w, h, c, data);
		stbi_image_free(data);
		return;
	}
	extension = ("tga");
	tp = name.compare(name.size() - extension.size(), extension.size(), extension) == 0;
	if (tp == 1)
	{
		stbi_write_tga(filename.c_str(), w, h, c, data);
		stbi_image_free(data);
		return;
	}
}
#endif
