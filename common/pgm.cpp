/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : Used in different projects to handle PGM I/O
 To build use  : 
 ============================================================================
 */
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "pgm.h"

using namespace std;

//-------------------------------------------------------------------
PGMImage::PGMImage(char *fname)
{
   x_dim=y_dim=num_colors=0;
   pixels=NULL;
   
   FILE *ifile;
   ifile = fopen(fname, "rb");
   if(!ifile) return;

   char *buff = NULL;
   size_t temp;

   fscanf(ifile, "%*s %i %i %i", &x_dim, &y_dim, &num_colors);

   getline((char **)&buff, &temp, ifile); // eliminate CR-LF
   
   assert(x_dim >1 && y_dim >1 && num_colors >1);
   pixels = new unsigned char[x_dim * y_dim];
   fread((void *) pixels, 1, x_dim*y_dim, ifile);   
   
   fclose(ifile);
}
//-------------------------------------------------------------------
PGMImage::PGMImage(int x=100, int y=100, int col=16)
{
   num_colors = (col>1) ? col : 16;
   x_dim = (x>1) ? x : 100;
   y_dim = (y>1) ? y : 100;
   pixels = new unsigned char[x_dim * y_dim];
   memset(pixels, 0, x_dim * y_dim);
   assert(pixels);
}
//-------------------------------------------------------------------
PGMImage::~PGMImage()
{
  if(pixels != NULL)
     delete [] pixels;
  pixels = NULL;
}
//-------------------------------------------------------------------
bool PGMImage::write(char *fname)
{
   int i,j;
   FILE *ofile;
   ofile = fopen(fname, "w+t");
   if(!ofile) return 0;

   fprintf(ofile,"P5\n%i %i\n%i\n",x_dim, y_dim, num_colors);
   fwrite(pixels, 1, x_dim*y_dim, ofile);
   fclose(ofile);
   return 1;
}

/* 
Funcion que dibuja las lineas en la imagen
image: imagen de entrada
w: ancho de la imagen
h: alto de la imagen
r: valor del radio
theta: valor del angulo
*/
void drawLine_(unsigned char * image, int w, int h, double r, double theta){
  int xCent = w / 2;
  int yCent = h / 2;

  double cosT = cos(theta);
  double sinT = sin(theta);

   if (fabs(sinT) > 0.5) { // Dibujar en función de x si sinT es grande
       for (int x = 0; x < w; x++) {
           double y = (r - (x - xCent) * cosT) / sinT;
           int yInt = yCent - (int)(y + 0.5);
           if (yInt >= 0 && yInt < h) {
               int idx = yInt * w + x;
               image[3 * idx] = 0;         // Componente R
               image[3 * idx + 1] = 200;   // Componente G
               image[3 * idx + 2] = 200;     // Componente B
           }
       }
   } else { // Dibujar en función de y si sinT es pequeño
       for (int y = 0; y < h; y++) {
           double x = (r - (yCent - y) * sinT) / cosT;
           int xInt = (int)(x + xCent + 0.5);
           if (xInt >= 0 && xInt < w) {
               int idx = y * w + xInt;
               image[3 * idx] = 0;         // Componente R
               image[3 * idx + 1] = 200;   // Componente G
               image[3 * idx + 2] = 200;     // Componente B
           }
       }
   }
}
/*
Funcion que permite detectar lineas basadas en un threshold
acc: arreglo de acumuladores
image: imagen de entrada
w: ancho de la imagen
h: alto de la imagen
rScale: escala de r
rMax: valor maximo de r
*/
void detectLines(int threshold, int *acc, unsigned char *image, int w, int h, double rScale, double rMax){
   const int degreeBins = 90;
   const int rBins = 100;
   const double radInc = 2 * M_PI / 180;
   std::vector<double> rArray;
   std::vector<double> anglesArray;
    for (int rIdx = 0; rIdx < rBins; rIdx++) {
        for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
            int idx = rIdx * degreeBins + tIdx;
            if (acc[idx] > threshold) {
               double r = rIdx * rScale - rMax;
               double theta = tIdx * radInc;
               rArray.push_back(r);
               anglesArray.push_back(theta);
            }
        }
    }
   for (int i = 0; i < rArray.size(); i++){
     drawLine_(image, w, h, rArray[i], anglesArray[i]);
   }
}
/*
Funcion para dibujar la imagen original
originalImage: imagen original
resultImage: imagen de salida
w: ancho de la imagen
h: alto de la imagen
*/
unsigned char* drawOriginalImage(PGMImage originalImage, unsigned char * resultImage, int w, int h){
  for (int i = 0; i < w * h; i++){
   unsigned char pixel = originalImage.pixels[i];
    resultImage[3*i] = pixel;
    resultImage[3*i + 1] = pixel;
    resultImage[3*i + 2] = pixel;
  }
  return resultImage;
}
/*
Funcion para dibujar la imagen con las lineas detectadas
*/
void drawImage(char *outputFileName, PGMImage image, int w, int h, int threshold, int *acc, double rScale, double rMax){
  unsigned char *resultImage = (unsigned char *)malloc(w * h * 3 * sizeof(unsigned char)); 
  if (!resultImage) {
    fprintf(stderr, "Failed to allocate memory for resultImage.\n");
    return;
  }

  resultImage = drawOriginalImage(image, resultImage, w, h); 

  // Safeguard for valid accumulator and threshold
  if (acc != nullptr && threshold >= 0) {
    detectLines(threshold, acc, resultImage, w, h, rScale, rMax);
  } else {
    fprintf(stderr, "Invalid accumulator or threshold.\n");
  }

  // Write the image using OpenCV
  cv::Mat img(h, w, CV_8UC3, resultImage);
  if (img.empty()) {
    fprintf(stderr, "Failed to create cv::Mat object.\n");
  } else {
    cv::imwrite(outputFileName, img);
  }

  free(resultImage); // Free the allocated memory
}
