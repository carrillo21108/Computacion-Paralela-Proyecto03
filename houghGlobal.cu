#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "common/pgm.h"

const int degreeInc = 2; 
const int degreeBins = 180 / degreeInc; 
const int rBins = 100; 
const float radInc = degreeInc * M_PI / 180;

int thresholdCalculus(int *cpuht) {
    int sum = 0, count = 0;
    for (int i = 0; i < degreeBins * rBins; i++) {
        sum += cpuht[i];
        if (cpuht[i] > 0) count++;
    }
    if (count == 0) {
        fprintf(stderr, "Error: No non-zero elements in the accumulator matrix.\n");
        return -1;
    }
    float avg = (float)sum / count;
    float stddev = 0;
    for (int i = 0; i < degreeBins * rBins; i++) {
        if (cpuht[i] > 0) stddev += pow(cpuht[i] - avg, 2);
    }
    stddev = sqrt(stddev / count);
    return (int)(avg + 2 * stddev);
}

void CPU_HoughTran(unsigned char *pic, int w, int h, int **acc) {
    float rMax = sqrt(1.0 * w * w + 1.0 * h * h) / 2;
    *acc = new int[rBins * degreeBins];
    memset(*acc, 0, sizeof(int) * rBins * degreeBins);

    int xCent = w / 2;
    int yCent = h / 2;
    float rScale = 2 * rMax / rBins;

    for (int i = 0; i < w; i++) {
        for (int j = 0; j < h; j++) {
            int idx = j * w + i;
            if (pic[idx] > 0) {
                int xCoord = i - xCent;
                int yCoord = yCent - j;
                float theta = 0;
                for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
                    float r = xCoord * cos(theta) + yCoord * sin(theta);
                    int rIdx = (r + rMax) / rScale;
                    (*acc)[rIdx * degreeBins + tIdx]++;
                    theta += radInc;
                }
            }
        }
    }
}

__global__ void GPU_HoughTran(unsigned char *pic, int w, int h, int *acc, float rMax, float rScale, float *d_Cos, float *d_Sin) {
    int gloID = blockIdx.x * blockDim.x + threadIdx.x;
    if (gloID >= w * h) return;

    int xCent = w / 2;
    int yCent = h / 2;

    int xCoord = gloID % w - xCent;
    int yCoord = yCent - gloID / w;

    if (pic[gloID] > 0) {
        for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
            float r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
            int rIdx = (r + rMax) / rScale;
            if (rIdx >= 0 && rIdx < rBins) {
                atomicAdd(&acc[rIdx * degreeBins + tIdx], 1);
            }
        }
    }
}

int main (int argc, char **argv)
{
  float *pcCos = (float *)malloc(sizeof(float) * degreeBins);
    float *pcSin = (float *)malloc(sizeof(float) * degreeBins);
    float rad = 0;
    for (int i = 0; i < degreeBins; i++) {
        pcCos[i] = cos(rad);
        pcSin[i] = sin(rad);
        rad += radInc;
    }

    float rMax = sqrt(1.0 * w * w + 1.0 * h * h) / 2;
    float rScale = 2 * rMax / rBins;

    cudaMemcpy(d_Cos, pcCos, sizeof(float) * degreeBins, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sin, pcSin, sizeof(float) * degreeBins, cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_in, sizeof(unsigned char) * w * h);
    
    cudaMalloc((void **)&d_hough, sizeof(int) * degreeBins * rBins);
    cudaMemcpy(d_in, inImg.pixels, sizeof(unsigned char) * w * h, cudaMemcpyHostToDevice);
    cudaMemset(d_hough, 0, sizeof(int) * degreeBins * rBins);

    int blockNum = ceil(w * h / 256);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    GPU_HoughTran<<<blockNum, 256>>>(d_in, w, h, d_hough, rMax, rScale, d_Cos, d_Sin);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("GPU Time: %f ms\n", milliseconds);

    int threshold = thresholdCalculus(h_hough);
    for (i = 0; i < degreeBins * rBins; i++) {
        if (cpuht[i] != h_hough[i])
            printf("Mismatch at %d: CPU=%d, GPU=%d\n", i, cpuht[i], h_hough[i]);
    }

    drawImage("houghGlobalGPU.jpg", inImg, w, h, threshold, h_hough, rScale, rMax);

    free(cpuht);
    free(pcCos);
    free(pcSin);
    free(h_hough);

    cudaFree(d_Cos);
    cudaFree(d_Sin);
    cudaFree(d_in);
    cudaFree(d_hough);

  return 0;
}