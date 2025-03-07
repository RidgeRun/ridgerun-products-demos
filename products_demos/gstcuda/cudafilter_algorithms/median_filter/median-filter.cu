/*
 * Copyright (C) 2018 RidgeRun, LLC (http://www.ridgerun.com)
 * All Rights Reserved.
 *
 * The contents of this software are proprietary and confidential to RidgeRun,
 * LLC.  No part of this program may be photocopied, reproduced or translated
 * into another programming language without prior written consent of
 * RidgeRun, LLC.  The user is free to modify the source code after obtaining
 * a software license from RidgeRun.  All source code changes must be provided
 * back to RidgeRun without any encumbrance.
 */

#undef __noinline__
#include <stdio.h>
#include <sys/cuda/filteralgorithm.hpp>
#define __noinline__ __attribute__((noinline))

// M(row, col) = *(M.data + row * M.pitch + col)
typedef struct {
    int width;
    int height;
    int pitch; 
    unsigned char * data;
} Matrix;

typedef struct {
    Matrix Y;
    Matrix U;
    Matrix V;
} YUVImage;

// Get a matrix element
__device__ unsigned char get_element(const Matrix A, int row, int col)
{
  return A.data[row * A.pitch + col];
}
// Set a matrix element
__device__ void set_element(Matrix A, int row, int col, unsigned char value)
{
  A.data[row * A.pitch + col] = value;
}
// Copy element from matrix A to matrix B
__device__ void copy_element(const Matrix A, Matrix B, int row, int col)
{
  B.data[row * B.pitch + col] = A.data[row * A.pitch + col];
}
// Sort only the upper half of an array
__device__ void half_bubble_sort(unsigned char *A, int N)
{
  int swap = 1;
  for(int i = 0; (i < N/2) && swap; i++){
    swap = 0;
    for (int j=0; j < N-1-i; j++){
      if (A[j+1] < A[j]){
        int temp = A[j];
        A[j] = A[j+1];
        A[j+1] = temp;
        swap = 1;
      }
    }
  }
}


/* 
 * Mask size used for separable filter (in_place)
 * TX1 GPU memory restrains this to: [1,47]
 */
#define MASK_S 7

/* 
 * Mask size used for regular filter (not_in_place)
 * TX1 GPU memory restrains this to: [1,5]
 */
#define MASK 3

/*
 * Thread array dimmensions: THREADS x THREADS
 * 32 is the optimal size because it uses all 1024 threads of the SM
 */
#define THREADS 32

__global__ void
cuda_accel_medianfilter_not_ip (YUVImage in, YUVImage out)
{
  __shared__ unsigned char A[THREADS][THREADS][MASK*MASK];
  // x and y positions on the image assigned to the current thread
  int xidx = blockIdx.x*THREADS+threadIdx.x;
  int yidx = blockIdx.y*THREADS+threadIdx.y;

  if (xidx<in.Y.width && yidx<in.Y.height){
    // Copy U V
    if(!(threadIdx.y & 1 || threadIdx.x & 1)){
      copy_element(in.U, out.U, yidx/2, xidx/2);
      copy_element(in.V, out.V, yidx/2, xidx/2);
    }
    // Median filter
    if (MASK/2<=xidx && xidx<in.Y.width-MASK/2 && MASK/2<=yidx && yidx<in.Y.height-MASK/2){
      for(int i=0; i<MASK; i++){
        for(int j=0; j<MASK; j++){
          A[threadIdx.y][threadIdx.x][i*MASK+j] = get_element(in.Y, yidx-MASK/2+i, xidx-MASK/2+j);
        }
      }
      half_bubble_sort(A[threadIdx.y][threadIdx.x], MASK*MASK);
      set_element(out.Y, yidx, xidx, A[threadIdx.y][threadIdx.x][MASK*MASK/2]);
    }
    else{
      copy_element(in.Y, out.Y, yidx, xidx);
    }
  }
}

__global__ void
cuda_accel_medianfilter_ip (YUVImage in)
{
  __shared__ unsigned char A[THREADS][THREADS][MASK_S];
  // x and y positions on the image assigned to the current thread
  int xidx = blockIdx.x*THREADS+threadIdx.x;
  int yidx = blockIdx.y*THREADS+threadIdx.y;

  if (xidx<in.Y.width && yidx<in.Y.height){
    // Median filter
    if (MASK_S/2<=xidx && xidx<in.Y.width-MASK_S/2 && MASK_S/2<=yidx && yidx<in.Y.height-MASK_S/2){
      // sort rows
      for(int i=0; i<MASK_S; i++){
        A[threadIdx.y][threadIdx.x][i] = get_element(in.Y, yidx, xidx-MASK_S/2+i);
      }
      half_bubble_sort(A[threadIdx.y][threadIdx.x], MASK_S);
      set_element(in.Y, yidx, xidx, A[threadIdx.y][threadIdx.x][MASK_S/2]);
      // fence
      __syncthreads();
      // sort cols
      for(int i=0; i<MASK_S; i++){
        A[threadIdx.y][threadIdx.x][i] = get_element(in.Y, yidx-MASK_S/2+i, xidx);
      }
      half_bubble_sort(A[threadIdx.y][threadIdx.x], MASK_S);
      set_element(in.Y, yidx, xidx, A[threadIdx.y][threadIdx.x][MASK_S/2]);
    }
  }
}

bool check_resolution_match (const GstCudaData * input_buffer, GstCudaData * output_buffer)
{
  if ((input_buffer->channels[0].width != output_buffer->channels[0].width) ||
      (input_buffer->channels[1].width != output_buffer->channels[1].width) ||
      (input_buffer->channels[2].width != output_buffer->channels[2].width) ||
      (input_buffer->channels[0].height != output_buffer->channels[0].height) ||
      (input_buffer->channels[1].height != output_buffer->channels[1].height) ||
      (input_buffer->channels[2].height != output_buffer->channels[2].height)) {
    return false;
  }
  else {
    return true;
  }
}

bool gst_cuda_data_to_yuvimage (const GstCudaData * cudaData, YUVImage * ret){
  ret->Y.width  = cudaData->channels[0].width;
  ret->Y.height = cudaData->channels[0].height;
  ret->Y.pitch  = cudaData->channels[0].pitch; 
  ret->Y.data   = (unsigned char *) cudaData->channels[0].data;
  ret->U.width  = cudaData->channels[1].width;
  ret->U.height = cudaData->channels[1].height;
  ret->U.pitch  = cudaData->channels[1].pitch; 
  ret->U.data   = (unsigned char *) cudaData->channels[1].data;
  ret->V.width  = cudaData->channels[2].width;
  ret->V.height = cudaData->channels[2].height;
  ret->V.pitch  = cudaData->channels[2].pitch; 
  ret->V.data   = (unsigned char *) cudaData->channels[2].data;
  return true;
}

bool cuda_medianfilter_not_ip (const GstCudaData * input_buffer,
		GstCudaData * output_buffer)
{
  cudaError_t cudaErr;
  cudaStream_t *stream;
  
  YUVImage in, out;

  if (!check_resolution_match (input_buffer, output_buffer)) {
    printf ("Error: Mismatch between input and output buffer size\n");
    return false;
  }

  int nBlocksx = ceil(input_buffer->channels[0].width/(THREADS*1.0));
  int nBlocksy = ceil(input_buffer->channels[0].height/(THREADS*1.0));
  dim3 grid_dimen = dim3 (nBlocksx, nBlocksy, 1);
  dim3 block_dimen = dim3 (THREADS, THREADS, 1);
  
  gst_cuda_data_to_yuvimage(input_buffer, &in);
  gst_cuda_data_to_yuvimage(output_buffer, &out);

  stream = (cudaStream_t*)input_buffer->stream;

  cuda_accel_medianfilter_not_ip  <<< grid_dimen, block_dimen, 0, *stream >>> (in, out);

  cudaErr = cudaGetLastError ();

  if (cudaSuccess != cudaErr) {
    printf ("CUDA kernel Error \n");
    return false;
  }

  cudaStreamSynchronize (*stream);

  cudaErr = cudaGetLastError ();

  if (cudaSuccess != cudaErr) {
    printf ("CUDA sync Error: %d\n", cudaErr);
    return false;
  }

  return true;
}

bool cuda_medianfilter_ip (GstCudaData * io_buffer)
{
  cudaError_t cudaErr;
  cudaStream_t *stream;

  YUVImage in;

  int nBlocksx = ceil(io_buffer->channels[0].width/(THREADS*1.0));
  int nBlocksy = ceil(io_buffer->channels[0].height/(THREADS*1.0));
  dim3 grid_dimen = dim3 (nBlocksx, nBlocksy, 1);
  dim3 block_dimen = dim3 (THREADS, THREADS, 1);

  gst_cuda_data_to_yuvimage(io_buffer, &in);

  stream = (cudaStream_t*)io_buffer->stream;

  cuda_accel_medianfilter_ip  <<< grid_dimen, block_dimen, 0, *stream >>> (in);

  cudaErr = cudaGetLastError ();

  if (cudaSuccess != cudaErr) {
    printf ("CUDA kernel Error \n");
    return false;
  }

  cudaStreamSynchronize (*stream);

  cudaErr = cudaGetLastError ();

  if (cudaSuccess != cudaErr) {
    printf ("CUDA sync Error: %d\n", cudaErr);
    return false;
  }

  return true;
}

class Example : public Gst::Cuda::Algorithm::Filter
{
public:
  bool open () override {
    //Put here the initialization logic
    return true;
  }

  bool close () override {
    //Put here the finalization logic
    return true;
  }

  bool process (const GstCudaData &input_buffer, GstCudaData &output_buffer) override {
    //Put here the process logic (This should be used when in_place is not configured)
    return cuda_medianfilter_not_ip (&input_buffer, &output_buffer);
  }

  bool process_ip (GstCudaData &io_buffer) override {
    //Put here the process logic (This should be used when in_place is configured)
    return cuda_medianfilter_ip (&io_buffer);
  }
};

Gst::Cuda::Algorithm::Filter *
factory_make ()
{
  return new Example ();
}
