#include "ex1.h"
using namespace std;

const int img_size = SIZE * SIZE * CHANNELS;
__device__ void prefixSum(int arr[], int size, int tid, int threads) {
    int increment;
    const auto is_active = tid < size;
    for (int stride = 1; stride<size; stride*=2) {
        if (tid >= stride && is_active) {
            increment = arr[tid - stride];
        }
        __syncthreads();
        if (tid >= stride && is_active) {
            arr[tid] += increment;
        }
        __syncthreads();
    }
}

__device__ void argmin(int arr[], int len, int tid, int threads) {
    int halfLen = len / 2;
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

__device__ void zero_array(int* histograms, int size=CHANNELS*LEVELS) {
    auto hist_flat = (int*) histograms;
    const int tid = threadIdx.x;
    const int threads = blockDim.x;
    for(int i = tid; i < size; i+=threads) {
        hist_flat[i] = 0;
    }
}

__device__ void colorHist(uchar img[][CHANNELS], int histograms[][LEVELS]) {
    const int pic_size = SIZE * SIZE;
    const int tid = threadIdx.x;
    const int threads = blockDim.x;

    for (int i = tid; i < 3*pic_size; i+=threads) {
        const int color = i%3;
        const int pixel = i/3;
        assert(pixel < pic_size);
        atomicAdd(&histograms[color][img[pixel][color]], 1);
    }
}

__device__ void performMapping(int maps[][LEVELS], uchar targetImg[][CHANNELS], uchar resultImg[][CHANNELS]){
    int pixels = SIZE * SIZE;
    const int tid = threadIdx.x;
    const int threads = blockDim.x;
    for (int i = tid; i < pixels; i+= threads) {
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
    int tid = threadIdx.x;;
    int threads = blockDim.x;
    int bid = blockIdx.x;
    __shared__ int deleta_cdf_row[LEVELS];
    __shared__ int map_cdf[CHANNELS][LEVELS];
    __shared__ int histogramsShared_target[CHANNELS][LEVELS];
    __shared__ int histogramsShared_refrence[CHANNELS][LEVELS];
    zero_array((int*)histogramsShared_target,   CHANNELS * LEVELS);
    zero_array((int*)histogramsShared_refrence, CHANNELS * LEVELS);
    zero_array((int*)map_cdf,                   CHANNELS * LEVELS);
    zero_array((int*)deleta_cdf_row,            LEVELS);

    auto target   = (uchar(*)[CHANNELS]) &targets[  bid * img_size];
    auto refrence = (uchar(*)[CHANNELS]) &refrences[bid * img_size];
    auto result   = (uchar(*)[CHANNELS]) &results[  bid * img_size];

    colorHist(target, histogramsShared_target);
    colorHist(refrence, histogramsShared_refrence);
    __syncthreads();

    for(int c=0; c < CHANNELS; c++)
    {   
        prefixSum(histogramsShared_target[c],LEVELS, threadIdx.x, blockDim.x);
        prefixSum(histogramsShared_refrence[c], LEVELS, threadIdx.x, blockDim.x);
        __syncthreads();

        for (int i = 0; i < LEVELS; i+=1) {
            for (int j = tid; j < LEVELS; j+=threads) {
                deleta_cdf_row[j] = abs(histogramsShared_target[c][i]-histogramsShared_refrence[c][j]);
            }
            __syncthreads();
            argmin(deleta_cdf_row, LEVELS, threadIdx.x, blockDim.x);
            __syncthreads();

            map_cdf[c][i] = deleta_cdf_row[1];

            __syncthreads();
        }
        __syncthreads();
    }          

    //Preform Map
    performMapping(map_cdf, target, result); 
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
    //allocate GPU memory for a single input image and a single output image
    CUDA_CHECK( cudaMalloc((void**)&(context->target_single),   img_size) ); 
    CUDA_CHECK( cudaMalloc((void**)&(context->refrence_single), img_size) ); 
    CUDA_CHECK( cudaMalloc((void**)&(context->result_single),   img_size) ); 
    return context;
}

/* Process all the images in the given host array and return the output in the
 * provided output host array */
void task_serial_process(struct task_serial_context *context, uchar *images_target, uchar *images_refrence, uchar *images_result)
{
    for(int i = 0; i < N_IMAGES; i++) {
        CUDA_CHECK( cudaMemcpy(context->target_single,   images_target,   img_size, cudaMemcpyHostToDevice) );
        CUDA_CHECK( cudaMemcpy(context->refrence_single, images_refrence, img_size, cudaMemcpyHostToDevice) );
        process_image_kernel<<<1,256>>>(context->target_single, context->refrence_single, context->result_single);
        CUDA_CHECK( cudaMemcpy(images_result, context->result_single, img_size, cudaMemcpyDeviceToHost) );
        images_target   += img_size;
        images_refrence += img_size;
        images_result   += img_size;
    }
}

/* Release allocated resources for the task-serial implementation. */
void task_serial_free(struct task_serial_context *context)
{
    cudaFree(context->refrence_single);
    cudaFree(context->target_single);
    cudaFree(context->result_single);
    free(context);
}

/* Bulk GPU context struct with necessary CPU / GPU pointers to process all the images */
struct gpu_bulk_context {
    // define bulk-GPU memory buffers
    uchar *target_single   = nullptr;
    uchar *refrence_single = nullptr;
    uchar *result_single   = nullptr;
};

/* Allocate GPU memory for all the input and output images.
 * Returns: allocated and initialized gpu_bulk_context. */
struct gpu_bulk_context *gpu_bulk_init()
{
    auto context = new gpu_bulk_context;

    // allocate GPU memory for all input images and all output images
    CUDA_CHECK( cudaMalloc((void**)&(context->target_single),  N_IMAGES * img_size) );
    CUDA_CHECK( cudaMalloc((void**)&(context->refrence_single),N_IMAGES * img_size) );
    CUDA_CHECK( cudaMalloc((void**)&(context->result_single),  N_IMAGES * img_size) );
    return context;
}

/* Process all the images in the given host array and return the output in the
 * provided output host array */
void gpu_bulk_process(struct gpu_bulk_context *context, uchar *images_target, uchar *images_refrence, uchar *images_result)
{
    // copy all input images from images_in to the GPU memory you allocated
    // invoke a kernel with N_IMAGES threadblocks, each working on a different image
    // copy output images from GPU memory to images_out
    CUDA_CHECK( cudaMemcpy(context->target_single,   images_target,   N_IMAGES * img_size, cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(context->refrence_single, images_refrence, N_IMAGES * img_size, cudaMemcpyHostToDevice) );
    process_image_kernel<<<N_IMAGES, 256>>>(context->target_single, context->refrence_single, context->result_single);
    CUDA_CHECK( cudaMemcpy(images_result, context->result_single, N_IMAGES * img_size, cudaMemcpyDeviceToHost) );
}

/* Release allocated resources for the bulk GPU implementation. */
void gpu_bulk_free(struct gpu_bulk_context *context)
{
    // free resources allocated in gpu_bulk_init
    cudaFree(context->refrence_single);
    cudaFree(context->target_single);
    cudaFree(context->result_single);
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
    zero_array((int*)histogramsShared);
    __syncthreads();

    int tid = threadIdx.x;
    int threads = blockDim.x;
    colorHist(img, histogramsShared);
    __syncthreads();


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
    
    __syncthreads();

    for(int i=tid; i<size; i+=threads){
        arr[i] = arrShared[i];
    }

    __syncthreads();
}

__global__ void performMappingWrapper(int maps[][LEVELS], uchar targetImg[][CHANNELS], uchar resultImg[][CHANNELS]){
    __syncthreads();
    performMapping(maps, targetImg, resultImg);
    __syncthreads();
}
