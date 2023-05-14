///////////////////////////////////////////////// DO NOT CHANGE ///////////////////////////////////////

#include "ex1.h"

void colorHist(uchar img[][CHANNELS], int pixelCount, int histograms[][LEVELS]){
    memset(histograms, 0, sizeof(int) * CHANNELS * LEVELS);
    for (int i = 0; i < pixelCount; i++) {
        uchar *rgbPixel = img[i];
        for (int j = 0; j < CHANNELS; j++){
            int *channelHist = histograms[j];
            channelHist[rgbPixel[j]] += 1;
        }
    }
}


void prefixSum(int arr[], int size, int res[]){
    int sum = 0;
    for (int i = 0; i < size; i++) {
        sum += arr[i];
        res[i] = sum;
    }
}

void imgHistCdf(uchar img[][CHANNELS], int pixelCount, uchar res[][LEVELS]){
    int histograms[CHANNELS][LEVELS];
    int tmp[CHANNELS][LEVELS];
    colorHist(img, pixelCount, histograms);
    for (int j = 0; j < CHANNELS; j++){
        int *channelHist = histograms[j];
        prefixSum(channelHist, LEVELS, tmp[j]);
    }
    for(int j = 0; j < CHANNELS; j++) {
        const int max = tmp[j][LEVELS -1];
        for(int i = 0; i < LEVELS; i++) {
            res[j][i] = (uchar) ((tmp[j][i]*256)/max);
        } 
    }
}

int argmin(int arr[], int size){
    int argmin = -1;
    int min = INT_MAX;
    for(int i = 0; i < size; i++){
        if (arr[i] < min){
            min = arr[i];
            argmin = i;
        }
    }
    return argmin;
}

void calculateMap(uchar targetCdf[], uchar refrenceCdf[],  uchar map[]){
    int diff[LEVELS][LEVELS];
    for(int i_ref = 0; i_ref < LEVELS; i_ref++){
        for(int i_tar = 0; i_tar < LEVELS; i_tar++){
            diff[i_tar][i_ref] = (int) abs(refrenceCdf[i_ref] - targetCdf[i_tar]);
        }
    }

    for(int row = 0; row < LEVELS; row++){
        map[row] = (uchar) argmin(diff[row], LEVELS);
    }
}

void performMapping(uchar maps[][LEVELS], uchar targetImg[][CHANNELS], uchar resultImg[][CHANNELS], int width, int height){
    int pixels = width * height;
    for (int i = 0; i < pixels; i++) {
        uchar *inRgbPixel = targetImg[i];
        uchar *outRgbPixel = resultImg[i];
        for (int j = 0; j < CHANNELS; j++){
            uchar *mapChannel = maps[j];
            outRgbPixel[j] = mapChannel[inRgbPixel[j]];
        }
    }
}

void cpu_process(uchar targetImg[][CHANNELS], uchar refrenceImg[][CHANNELS],  uchar outputImg[][CHANNELS], int width, int height) {
    uchar targetCdf[CHANNELS][LEVELS];
    uchar refrenceCdf[CHANNELS][LEVELS];
    uchar maps[CHANNELS][LEVELS];
    int pixelCount = width * height;
    imgHistCdf(targetImg, pixelCount, targetCdf);
    imgHistCdf(refrenceImg, pixelCount, refrenceCdf);
    
    for (int i = 0; i < CHANNELS; i++){
        uchar *refrencetChannelCdf = refrenceCdf[i];
        uchar *targetChannelCdf = targetCdf[i];
        uchar *mapChannel = maps[i];
        calculateMap(targetChannelCdf, refrencetChannelCdf, mapChannel);
    }

    performMapping(maps, targetImg, outputImg, width, height);
}


