#include "ex1.h"
using namespace std;
__device__ void prefixSum(int arr[], int size, int tid, int threads) {
    int increment;
    if(tid >= size)
        return;
    for (int stride = 1; stride<threads; stride*=2) {
        if (tid >= stride) {
            increment = arr[tid - stride];
        }
        __syncthreads();
        if (tid >= stride) {
            arr[tid] += increment;
        }
        __syncthreads();
    }

}

__device__ void argmin(int arr[], int len, int tid, int threads) {
    int halfLen = len / 2;    
    assert(threads == halfLen);
    assert(tid < threads);
    bool firstIteration = true;
    int prevHalfLength = 0;
    while (halfLen > 0) {
        if(tid < halfLen){
            if(arr[tid] == arr[tid + halfLen]){ //a corenr case
                int lhsIdx = tid;
                int rhdIdx = tid + halfLen;
                int lhsOriginalIdx = firstIteration ? lhsIdx : arr[prevHalfLength + lhsIdx];
                int rhsOriginalIdx = firstIteration ? rhdIdx : arr[prevHalfLength + rhdIdx];
                arr[tid + halfLen] = lhsOriginalIdx < rhsOriginalIdx ? lhsOriginalIdx : rhsOriginalIdx;
            }
            else { //the common case
                bool isLhsSmaller = (arr[tid] < arr[tid + halfLen]);
                int idxOfSmaller = isLhsSmaller * tid + (!isLhsSmaller) * (tid + halfLen);
                int smallerValue = arr[idxOfSmaller];
                int origIdxOfSmaller = firstIteration * idxOfSmaller + (!firstIteration) * arr[prevHalfLength + idxOfSmaller];
                arr[tid] = smallerValue;
                arr[tid + halfLen] = origIdxOfSmaller;
            }
        }
        __syncthreads();
        firstIteration = false;
        prevHalfLength = halfLen;
        halfLen /= 2;
    }
}

__device__ void colorHist(uchar img[][CHANNELS], int histograms[][LEVELS]) {

    const int pic_size = SIZE * SIZE;
    auto hist_flat = (int*) histograms;
    const int tid = threadIdx.x;
    const int threads = blockDim.x;

    __syncthreads();
    for(int i = tid; i < 3*LEVELS; i+=threads) {
        hist_flat[i] = 0;
    }
    __syncthreads();

    for (int i = tid; i < 3*pic_size; i+=threads) {
        const int color = i%3;
        const int pixel = i/3;
        assert(pixel < pic_size);
        atomicAdd(&histograms[color][img[pixel][color]], 1);
    }    
    // old version
    // {
    // const int r = 0;
    // const int g = 1;
    // const int b = 2;
    // for (int i = tid; i < pic_size; i+=threads) {
    //     atomicAdd(&histograms[r][img[i][r]], 1);
    //     atomicAdd(&histograms[g][img[i][g]], 1);
    //     atomicAdd(&histograms[b][img[i][b]], 1);
    // }
    __syncthreads();

}

__device__ void performMapping(int maps[][LEVELS], uchar targetImg[][CHANNELS], uchar resultImg[][CHANNELS]){
    int pixels = SIZE * SIZE;
    const int tid = threadIdx.x;
    const int threads = blockDim.x;
    __syncthreads();
    for (int i = tid; i < pixels; i+= threads) {
        uchar *inRgbPixel = targetImg[i];
        uchar *outRgbPixel = resultImg[i];
        for (int j = 0; j < CHANNELS; j++){
            int *mapChannel = maps[j];
            outRgbPixel[j] = mapChannel[inRgbPixel[j]];
        }
    }    
    __syncthreads();
}
__device__ void  create_map(int cdf_1[][LEVELS],int cdf_2[][LEVELS],int abs_cdf[][LEVELS]){
    int tid = threadIdx.x;
    int threads = blockDim.x;
    __syncthreads();
    for (int i = tid; i < LEVELS; i+=threads) {
        for (int j = 0; j < CHANNELS; j++){
            abs_cdf[j][i] = abs(cdf_1[j][i]-cdf_2[j][i]);
        }
    }     
    __syncthreads();
}

__device__
int argmin_cpu(int arr[], int size){
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

__device__
void performMapping_cpu(int maps[][LEVELS], uchar targetImg[][CHANNELS], uchar resultImg[][CHANNELS]){
    int pixels = SIZE * SIZE;
    for (int i = 0; i < pixels; i++) {
        uchar *inRgbPixel = targetImg[i];
        uchar *outRgbPixel = resultImg[i];
        for (int j = 0; j < CHANNELS; j++){
            int *mapChannel = maps[j];
            outRgbPixel[j] = mapChannel[inRgbPixel[j]];
        }
    }
}


__global__
void process_image_kernel(uchar *targets, uchar *refrences, uchar *results) {
    // TODO   
    int tid = threadIdx.x;;
    int threads = blockDim.x;
    //int img_size = CHANNELS*SIZE * SIZE;
    
    int abs_cdf[CHANNELS][LEVELS];
    int hist_target[CHANNELS][LEVELS];
    int hist_refrence[CHANNELS][LEVELS];
   
    auto target = (uchar(*)[CHANNELS]) targets;
    auto refrence = (uchar(*)[CHANNELS]) refrences;
    auto result = (uchar(*)[CHANNELS]) results;
     //find hist of images
    __shared__ int histogramsShared_target[CHANNELS][LEVELS];
    __shared__ int histogramsShared_refrence[CHANNELS][LEVELS];

    __syncthreads(); 
    
    colorHist(target, histogramsShared_target);
    
    __syncthreads();    

    colorHist(refrence, histogramsShared_refrence);

    __syncthreads();

    for(int i=0;i<CHANNELS;i++)
    {   
        prefixSum(histogramsShared_target[i], LEVELS, threadIdx.x, blockDim.x);
        __syncthreads();

        prefixSum(histogramsShared_refrence[i], LEVELS, threadIdx.x, blockDim.x);
        __syncthreads();

        for(int i = tid; i < CHANNELS * LEVELS; i+=threads){
            ((int*)hist_target)[i] = ((int*)histogramsShared_target)[i];
            __syncthreads();
            ((int*)hist_refrence)[i] = ((int*)histogramsShared_refrence)[i];
            __syncthreads();
        }
    }
    //Create Map
    create_map(hist_target, hist_refrence, abs_cdf);

    __syncthreads();
    //argmin
    for(int i=0;i<CHANNELS;i++) {//i'm not sure
        // if (tid < LEVELS / 2) {
        //     argmin(abs_cdf[i], LEVELS, tid, LEVELS / 2);
        // }
        if (tid == 0) { //for debugging
            argmin_cpu(abs_cdf[i], LEVELS);
        }
    }
    __syncthreads();
    // for (int i = tid; i < LEVELS; i+=threads) {
    //     printf(" %d --- %d \n", i, (int) abs_cdf[0][i]);
    // }
    

    //Preform Map
    performMapping(abs_cdf, target, result);
    __syncthreads();

    for (int i = tid; i < LEVELS; i+=threads) {
        printf(" %d --- %d \n", i, (int) result[i][0]);
    }
    
    __syncthreads();  
}


/* Task serial context struct with necessary CPU / GPU pointers to process a single image */
struct task_serial_context {
    uchar *target_single   = nullptr;
    uchar *refrence_single = nullptr;
    uchar *result_single   = nullptr;
};

/* Allocate GPU memory for a single input image and a single output image.
 * 
 * Returns: allocated and initialized task_serial_context. */
struct task_serial_context* task_serial_init()
{
    auto context = new task_serial_context;
    //TODO: allocate GPU memory for a single input image and a single output image
    CUDA_CHECK( cudaMalloc((void**)&context->target_single,SIZE*SIZE*LEVELS) ); 
    CUDA_CHECK( cudaMalloc((void**)&context->refrence_single,SIZE*SIZE*LEVELS) ); 
    CUDA_CHECK( cudaMalloc((void**)&context->result_single,SIZE*SIZE*LEVELS) ); 
    return context;
}

/* Process all the images in the given host array and return the output in the
 * provided output host array */
void task_serial_process(struct task_serial_context *context, uchar *images_target, uchar *images_refrence, uchar *images_result)
{
    //TODO: in a for loop:
    //   1. copy the relevant image from images_in to the GPU memory you allocated
    //   2. invoke GPU kernel on this image
    //   3. copy output from GPU memory to relevant location in images_out_gpu_serial
    int size_img = sizeof(uchar)*LEVELS*SIZE*SIZE;
    for (int i = 0; i < 5; i++)
    {
        CUDA_CHECK( cudaMemcpy(context->target_single,  images_target  +(i*(size_img)), size_img, cudaMemcpyHostToDevice) );
        CUDA_CHECK( cudaMemcpy(context->refrence_single,images_refrence+(i*(size_img)), size_img, cudaMemcpyHostToDevice) );
        process_image_kernel<<<1,1024>>>(context->target_single, context->refrence_single, context->result_single);
        CUDA_CHECK( cudaMemcpy(images_result+(i*(size_img)), context->result_single, size_img, cudaMemcpyDeviceToHost) );
    }    

}

/* Release allocated resources for the task-serial implementation. */
void task_serial_free(struct task_serial_context *context)
{
    cudaFree((void**)context->refrence_single);
    cudaFree((void**)context->target_single);
    cudaFree((void**)context->result_single);
    free(context);
}

/* Bulk GPU context struct with necessary CPU / GPU pointers to process all the images */
struct gpu_bulk_context {
    // TODO define bulk-GPU memory buffers
};

/* Allocate GPU memory for all the input and output images.
 * Returns: allocated and initialized gpu_bulk_context. */
struct gpu_bulk_context *gpu_bulk_init()
{
    auto context = new gpu_bulk_context;

    //TODO: allocate GPU memory for all input images and all output images

    return context;
}

/* Process all the images in the given host array and return the output in the
 * provided output host array */
void gpu_bulk_process(struct gpu_bulk_context *context, uchar *images_target, uchar *images_refrence, uchar *images_result)
{
    //TODO: copy all input images from images_in to the GPU memory you allocated
    //TODO: invoke a kernel with N_IMAGES threadblocks, each working on a different image
    //TODO: copy output images from GPU memory to images_out

}

/* Release allocated resources for the bulk GPU implementation. */
void gpu_bulk_free(struct gpu_bulk_context *context)
{
    //TODO: free resources allocated in gpu_bulk_init
    free(context);
}


/********************************************************
**  the following waappers are needed for unit testing.
********************************************************/

__global__ void argminWrapper(int arr[], int size){
    argmin(arr, size, threadIdx.x, blockDim.x);
}

__global__ void colorHistWrapper(uchar img[][CHANNELS], int histograms[][LEVELS]){
    __shared__ int histogramsShared[CHANNELS][LEVELS];    
    int tid = threadIdx.x;
    int threads = blockDim.x;
    colorHist(img, histogramsShared);
    for(int i = tid; i < CHANNELS * LEVELS; i+=threads){
        ((int*)histograms)[i] = ((int*)histogramsShared)[i];
    }
    __syncthreads();

}

__global__ void prefixSumWrapper(int arr[], int size){
    __shared__ int arrShared[LEVELS];
    int tid = threadIdx.x;
    int threads = blockDim.x;

    for(int i=tid; i<size; i+=threads){
        arrShared[i] = arr[i];
    }

    __syncthreads();

    prefixSum(arrShared, size, threadIdx.x, blockDim.x);
    
    for(int i=tid; i<size; i+=threads){
        arr[i] = arrShared[i];
    }

    __syncthreads();
}

__global__ void performMappingWrapper(int maps[][LEVELS], uchar targetImg[][CHANNELS], uchar resultImg[][CHANNELS]){
    performMapping(maps, targetImg, resultImg);
}
