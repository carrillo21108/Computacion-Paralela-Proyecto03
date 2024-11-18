#pragma once
#ifndef PGM_H
#define PGM_H
#include <opencv2/opencv.hpp>

class PGMImage
{
 public:
   PGMImage(char *);
   PGMImage(int x, int y, int col);
   ~PGMImage();
   bool write(char *);
		   
   int x_dim;
   int y_dim;
   int num_colors;
   unsigned char *pixels;
};
void drawImage(char *outputFileName, unsigned char* pixels, int w, int h, int threshold, int *acc, double rScale, double rMax);
#endif