/*
 * Copyright (C) 2017 RidgeRun, LLC (http://www.ridgerun.com)
 * All Rights Reserved.
 *
 * The contents of this software are proprietary and confidential to RidgeRun,
 * LLC.  No part of this program may be photocopied, reproduced or translated
 * into another programming language without prior written consent of
 * RidgeRun, LLC.  The user is free to modify the source code after obtaining
 * a software license from RidgeRun.  All source code changes must be provided
 * back to RidgeRun without any encumbrance.
 */

#include <iostream>
#include <math.h>

#undef __noinline__
#include <sys/cuda/filteralgorithm.hpp>
#define __noinline__ __attribute__((noinline))

__global__ void
pinhole_gpu (unsigned char *output,
	     unsigned char *input,
	     int cols, int rows, int stride,
	     float P00, float P01, float P02,
	     float P10, float P11, float P12,
	     float P20, float P21, float P22)
{

  int row = blockIdx.y*blockDim.y + threadIdx.y;
  int col = blockIdx.x*blockDim.x + threadIdx.x;

  /* Avoid over-processing on non-matching execution configurations */
  if (row >= rows || col >= cols) {
    return;
  }
  
  int cy = rows/2;
  int cx = cols/2;

  /* Traslate the center of the image to the origin */
  int x = col - cx;
  int y = row - cy;

  /* Compute 3D rotation */
  float u_ = P00*x + P01*y + P02;
  float v_ = P10*x + P11*y + P12;
  float w_ = P20*x + P21*y + P22;

  /* Project back to 2D:
   * - Ugly trucation-based nearest neighbor interpolation
   */
  int u = u_/w_;
  int v = v_/w_;

  /* Return back to left coordinate system */
  u += cx;
  v += cy;
      
  int idxo = row*stride + col;
  int idxi = v*stride + u;

  if (v >= 0 && v < rows && u >= 0 && u < cols){
    output[idxo] = input[idxi];
  }
}

__global__ void
clear_color (unsigned char *output, int cols, int rows, int stride)

{
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  int col = blockIdx.x*blockDim.x + threadIdx.x;
  int idx = row*stride + col;

  /* Avoid over-processing on non-matching execution configurations */
  if (row >= rows || col >= cols) {
    return;
  }

  /* Gray out image */
  output[idx] = 128;
}

class Example : public Gst::Cuda::Algorithm::Filter
{
private:
  int ax;
  int ay;
  int az;

  float f;

  float to_radians (float degrees) {
    return degrees*M_PI/180;
  }
  
  bool find_inverse_mapping (float ax, float ay, float az, float f,
			     float &P00, float &P01, float &P02,
			     float &P10, float &P11, float &P12,
			     float &P20, float &P21, float &P22)
  {
    float P[3][3];

    ax = this->to_radians (ax);
    ay = this->to_radians (ay);
    az = this->to_radians (az);
    
    P[0][0] = cos(ay)*cos(az)*f;
    P[0][1] = ((-cos(ax)*sin(az)) - sin(ax)*sin(ay)*cos(az))*f;
    P[0][2] = 0;
    
    P[1][0] = cos(ay)*sin(az)*f;
    P[1][1] = (cos(ax)*cos(az) - sin(ax)*sin(ay)*sin(az))*f;
    P[1][2] = 0;

    P[2][0] = sin(ay);
    P[2][1] = sin(ax)*cos(ay);
    P[2][2] = f;

    float det =
      P[0][0] * (P[1][1] * P[2][2] - P[2][1] * P[1][2]) -
      P[0][1] * (P[1][0] * P[2][2] - P[1][2] * P[2][0]) +
      P[0][2] * (P[1][0] * P[2][1] - P[1][1] * P[2][0]);

    if (0 == det) {
      std::cerr << "Unable to compute inverse of singular matrix" << std::endl;
      return false;
    }
    
    float invdet = 1 / det;

    P00 = (P[1][1] * P[2][2] - P[2][1] * P[1][2]) * invdet;
    P01 = (P[0][2] * P[2][1] - P[0][1] * P[2][2]) * invdet;
    P02 = (P[0][1] * P[1][2] - P[0][2] * P[1][1]) * invdet;
    P10 = (P[1][2] * P[2][0] - P[1][0] * P[2][2]) * invdet;
    P11 = (P[0][0] * P[2][2] - P[0][2] * P[2][0]) * invdet;
    P12 = (P[1][0] * P[0][2] - P[0][0] * P[1][2]) * invdet;
    P20 = (P[1][0] * P[2][1] - P[2][0] * P[1][1]) * invdet;
    P21 = (P[2][0] * P[0][1] - P[0][0] * P[2][1]) * invdet;
    P22 = (P[0][0] * P[1][1] - P[1][0] * P[0][1]) * invdet;

    return true;
  }
  
  bool pinhole (const GstCudaData * input_buffer,
		GstCudaData * output_buffer)
  {
    cudaError_t cudaErr;
    cudaStream_t *stream = (cudaStream_t*)input_buffer->stream;
    int width = input_buffer->channels[0].width;
    int height = input_buffer->channels[0].height;
    int stride = input_buffer->channels[0].pitch;
    int width_u = input_buffer->channels[1].width;
    int height_u = input_buffer->channels[1].height;
    int stride_u = input_buffer->channels[1].pitch;
    int width_v = input_buffer->channels[2].width;
    int height_v = input_buffer->channels[2].height;
    int stride_v = input_buffer->channels[2].pitch;
    unsigned char * input = (unsigned char *)input_buffer->channels[0].data;
    unsigned char * output = (unsigned char *)output_buffer->channels[0].data;
    unsigned char * out_u = (unsigned char *)output_buffer->channels[1].data;
    unsigned char * out_v = (unsigned char *)output_buffer->channels[2].data;

    float max_threads = 32;
    dim3 grid_dimen = dim3 (ceil(width/max_threads), ceil(height/max_threads), 1);
    dim3 block_dimen = dim3 (max_threads, max_threads, 1);
    
    float P00 = 0, P01 = 0, P02 = 0;
    float P10 = 0, P11 = 0, P12 = 0;
    float P20 = 0, P21 = 0, P22 = 0;

    /* Cheap way to parametrize focal distance from image
       resolution */
    this->f = width > height ? width : height;
    
    /* Geometric transforms are made by iterating through the output
     * image pixels and finding the respective origin pixel from source
     * image with some sort of interpolation. As such we need to find
     * the inverse mapping of the given rotation angles.
     */
    this->find_inverse_mapping (this->ax, this->ay, this->az, this->f,
				P00, P01, P02,
				P10, P11, P12,
				P20, P21, P22);
    

    pinhole_gpu  <<< grid_dimen, block_dimen, 0, *stream >>>
      (output, input, width, height, stride,
       P00, P01, P02,
       P10, P11, P12,
       P20, P21, P22);

    grid_dimen = dim3 (ceil(width_u/max_threads), ceil(height_u/max_threads), 1);
    block_dimen = dim3 (max_threads, max_threads, 1);

    clear_color  <<< grid_dimen, block_dimen, 0, *stream >>> (out_u, width_u, height_u, stride_u);
    clear_color  <<< grid_dimen, block_dimen, 0, *stream >>> (out_v, width_v, height_v, stride_v);
    
    this->ax = (this->ax + 1) % 360;
    this->ay = (this->ay + 1) % 360;
    this->az = (this->az + 1) % 360;
    
    cudaErr = cudaGetLastError ();
    if (cudaSuccess != cudaErr) {
      std::cerr << "CUDA kernel Error" << std::endl;
      return false;
    }
    
    cudaStreamSynchronize (*stream);

    cudaErr = cudaGetLastError ();
    if (cudaSuccess != cudaErr) {
      std::cerr << "CUDA sync Error: " <<  cudaErr << std::endl;
      return false;
    }
    
    return true;
  }

public:
  bool open () override {
    this->ax = 0;
    this->ay = 0;
    this->az = 0;
    this->f = 0;

    return true;
  }

  bool close () override {
    //Put here the finalization logic
    return true;
  }

  bool process (const GstCudaData &input_buffer, GstCudaData &output_buffer) override {
    return pinhole (&input_buffer, &output_buffer);
  }

  bool process_ip (GstCudaData &io_buffer) override {
    return true;
  }
};

Gst::Cuda::Algorithm::Filter *
factory_make ()
{
  return new Example ();
}
