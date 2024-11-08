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
