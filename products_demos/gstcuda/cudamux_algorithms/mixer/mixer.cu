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

#undef __noinline__
#include <stdio.h>
#include <sys/cuda/muxalgorithm.hpp>
#define __noinline__ __attribute__((noinline))

__global__ void
cuda_accel_mixer_grey8_not_ip (unsigned char *InY1, unsigned char *InY2,
    unsigned char *OutY, int width, int height, int stride_y)
{
  /* Compute indexes */
  int yIndex = (2 * width * blockIdx.y) + (2 * threadIdx.x) +
      ((width / 2) * (blockIdx.x));

  int yIndexPitchOdd = yIndex + (2 * stride_y * blockIdx.y);
  int yIndexpitchEven = yIndexPitchOdd + stride_y;

  OutY[yIndex] = (InY1[yIndexPitchOdd] + InY2[yIndexPitchOdd]) / 2;
  OutY[yIndex + 1] = (InY1[yIndexPitchOdd + 1] + InY2[yIndexPitchOdd + 1]) / 2;
  OutY[yIndex + width] = (InY1[yIndexpitchEven + width] + InY2[yIndexpitchEven + width]) / 2;
  OutY[yIndex + width + 1] = (InY1[yIndexpitchEven + width + 1] + InY2[yIndexpitchEven + width + 1]) /2;
}

__global__ void
cuda_accel_mixer_grey8_ip (unsigned char *InY1, unsigned char *OutY,
    int width, int height, int stride_y)
{
  int tmp_OutY;

  /* Compute indexes */
  int yIndex = (2 * width * blockIdx.y) + (2 * threadIdx.x) +
      ((width / 2) * (blockIdx.x));

  int yIndexPitchOdd = yIndex + (2 * stride_y * blockIdx.y);
  int yIndexpitchEven = yIndexPitchOdd + stride_y;

  tmp_OutY = OutY[yIndexPitchOdd];
  OutY[yIndexPitchOdd] = (InY1[yIndexPitchOdd] + tmp_OutY)/2;
  tmp_OutY = OutY[yIndexPitchOdd + 1];
  OutY[yIndexPitchOdd + 1] = (InY1[yIndexPitchOdd + 1] + tmp_OutY)/2;
  tmp_OutY = OutY[yIndexpitchEven + width];
  OutY[yIndexpitchEven + width] = (InY1[yIndexpitchEven + width] + tmp_OutY)/2;
  tmp_OutY = OutY[yIndexpitchEven + width + 1];
  OutY[yIndexpitchEven + width + 1] = (InY1[yIndexpitchEven + width + 1] + tmp_OutY)/2;
}

__global__ void
cuda_accel_mixer_i420_not_ip (unsigned char *InY1, unsigned char *InU1,
    unsigned char *InV1, unsigned char *InY2, unsigned char *InU2,
    unsigned char *InV2, unsigned char *OutY, unsigned char *OutU,
    unsigned char *OutV, int width, int height, int stride_y, int stride_uv)
{
  /* Compute indexes */
  int yIndex = (2 * width * blockIdx.y) + (2 * threadIdx.x) +
      ((width / 2) * (blockIdx.x));

  int yIndexPitchOdd = yIndex + (2 * stride_y * blockIdx.y);
  int yIndexpitchEven = yIndexPitchOdd + stride_y;

  int uvIndex = ((width / 2) * blockIdx.y) +
      threadIdx.x + ((width / 4) * blockIdx.x);

  int uvIndexPitch = uvIndex + ((stride_uv) * blockIdx.y);

  OutU[uvIndex] = (InU1[uvIndexPitch] + InU2[uvIndexPitch]) / 2;
  OutV[uvIndex] = (InV1[uvIndexPitch] + InV2[uvIndexPitch]) / 2;

  OutY[yIndex] = (InY1[yIndexPitchOdd] + InY2[yIndexPitchOdd]) / 2;
  OutY[yIndex + 1] = (InY1[yIndexPitchOdd + 1] + InY2[yIndexPitchOdd + 1]) / 2;
  OutY[yIndex + width] = (InY1[yIndexpitchEven + width] + InY2[yIndexpitchEven + width]) / 2;
  OutY[yIndex + width + 1] = (InY1[yIndexpitchEven + width + 1] + InY2[yIndexpitchEven + width + 1]) /2;
}

__global__ void
cuda_accel_mixer_i420_ip (unsigned char *InY1, unsigned char *InU1,
    unsigned char *InV1, unsigned char *OutY, unsigned char *OutU,
    unsigned char *OutV, int width, int height, int stride_y, int stride_uv)
{
  int tmp_OutUV;
  int tmp_OutY;

  /* Compute indexes */
  int yIndex = (2 * width * blockIdx.y) + (2 * threadIdx.x) +
      ((width / 2) * (blockIdx.x));

  int yIndexPitchOdd = yIndex + (2 * stride_y * blockIdx.y);
  int yIndexpitchEven = yIndexPitchOdd + stride_y;

  int uvIndex = ((width / 2) * blockIdx.y) +
      threadIdx.x + ((width / 4) * blockIdx.x);

  int uvIndexPitch = uvIndex + ((stride_uv) * blockIdx.y);

  tmp_OutUV = OutU[uvIndexPitch];
  OutU[uvIndexPitch] = (InU1[uvIndexPitch] + tmp_OutUV)/2;
  tmp_OutUV = OutV[uvIndexPitch];
  OutV[uvIndexPitch] = (InV1[uvIndexPitch] + tmp_OutUV)/2;

  tmp_OutY = OutY[yIndexPitchOdd];
  OutY[yIndexPitchOdd] = (InY1[yIndexPitchOdd] + tmp_OutY)/2;
  tmp_OutY = OutY[yIndexPitchOdd + 1];
  OutY[yIndexPitchOdd + 1] = (InY1[yIndexPitchOdd + 1] + tmp_OutY)/2;
  tmp_OutY = OutY[yIndexpitchEven + width];
  OutY[yIndexpitchEven + width] = (InY1[yIndexpitchEven + width] + tmp_OutY)/2;
  tmp_OutY = OutY[yIndexpitchEven + width + 1];
  OutY[yIndexpitchEven + width + 1] = (InY1[yIndexpitchEven + width + 1] + tmp_OutY)/2;
}

bool check_resolution_match (std::vector<GstCudaData> input_buffers, GstCudaData * output_buffer)
{
  if (input_buffers.at(0).format == GST_CUDA_I420){
   return ((input_buffers.at(0).channels[0].width == output_buffer->channels[0].width) &&
           (input_buffers.at(0).channels[1].width == output_buffer->channels[1].width) &&
           (input_buffers.at(0).channels[2].width == output_buffer->channels[2].width) &&
           (input_buffers.at(0).channels[0].height == output_buffer->channels[0].height) &&
           (input_buffers.at(0).channels[1].height == output_buffer->channels[1].height) &&
           (input_buffers.at(0).channels[2].height == output_buffer->channels[2].height));
  } else if (input_buffers.at(0).format == GST_CUDA_GREY) {
   return (input_buffers.at(0).channels[0].width == output_buffer->channels[0].width) &&
	  (input_buffers.at(0).channels[0].height == output_buffer->channels[0].height);
  } else {
    return false;
  }
}

bool check_inputs_resolution_match (std::vector<GstCudaData> input_buffers)
{
  if (input_buffers.at(0).format == GST_CUDA_I420){
   return (input_buffers.at(0).channels[0].width == input_buffers.at(1).channels[0].width) &&
          (input_buffers.at(0).channels[1].width == input_buffers.at(1).channels[1].width) &&
          (input_buffers.at(0).channels[2].width == input_buffers.at(1).channels[2].width) &&
          (input_buffers.at(0).channels[0].height == input_buffers.at(1).channels[0].height) &&
          (input_buffers.at(0).channels[1].height == input_buffers.at(1).channels[1].height) &&
          (input_buffers.at(0).channels[2].height == input_buffers.at(1).channels[2].height);
  } else if (input_buffers.at(0).format == GST_CUDA_GREY) {
    return (input_buffers.at(0).channels[0].width == input_buffers.at(1).channels[0].width) &&
           (input_buffers.at(0).channels[0].height == input_buffers.at(1).channels[0].height);
  } else {
    return false;
  }
}

bool cuda_mixer_not_ip (std::vector<GstCudaData> input_buffers,
		GstCudaData * output_buffer)
{
  cudaError_t cudaErr;
  cudaStream_t *stream;

  if (!check_inputs_resolution_match (input_buffers)) {
    printf ("Error: Mismatch between input buffers size\n");
    return false;
  }

  if (!check_resolution_match (input_buffers, output_buffer)) {
    printf ("Error: Mismatch between input and output buffer size\n");
    return false;
  }

  dim3 grid_dimen = dim3 (2, (input_buffers.at(0).channels[0].height / 2), 1);
  dim3 block_dimen = dim3 ((input_buffers.at(0).channels[0].width / 4), 1, 1);
  int in0_padding_ch0 = input_buffers.at(0).channels[0].pitch - input_buffers.at(0).channels[0].width;
  int in0_padding_ch1 = input_buffers.at(0).channels[1].pitch - input_buffers.at(0).channels[1].width;
  int in1_padding_ch0 = input_buffers.at(1).channels[0].pitch - input_buffers.at(1).channels[0].width;
  int in1_padding_ch1 = input_buffers.at(1).channels[1].pitch - input_buffers.at(1).channels[1].width;

  if (in0_padding_ch0 != in1_padding_ch0) {
    printf ("Error: Mismatch between inputs ch0 (Y) padding\n");
    return false;
  }

  if (input_buffers.at(0).format == GST_CUDA_I420 && in0_padding_ch1 != in1_padding_ch1) {
    printf ("Error: Mismatch between inputs ch1 (UV) padding\n");
    return false;
  }

  stream = (cudaStream_t*)output_buffer->stream;

  switch(input_buffers.at(0).format){
    case GST_CUDA_I420:
      cuda_accel_mixer_i420_not_ip  <<< grid_dimen, block_dimen, 0, *stream >>> (
         (unsigned char *) input_buffers.at(0).channels[0].data,
         (unsigned char *) input_buffers.at(0).channels[1].data,
	 (unsigned char *) input_buffers.at(0).channels[2].data,
	 (unsigned char *) input_buffers.at(1).channels[0].data,
	 (unsigned char *) input_buffers.at(1).channels[1].data,
	 (unsigned char *) input_buffers.at(1).channels[2].data,
	 (unsigned char *) output_buffer->channels[0].data,
	 (unsigned char *) output_buffer->channels[1].data,
	 (unsigned char *) output_buffer->channels[2].data,
	 input_buffers.at(0).channels[0].width,
	 input_buffers.at(0).channels[0].height,
	 in0_padding_ch0, in0_padding_ch1);
      break;
    case GST_CUDA_GREY:
      cuda_accel_mixer_grey8_not_ip  <<< grid_dimen, block_dimen, 0, *stream >>> (
         (unsigned char *) input_buffers.at(0).channels[0].data,
	 (unsigned char *) input_buffers.at(1).channels[0].data,
	 (unsigned char *) output_buffer->channels[0].data,
	 input_buffers.at(0).channels[0].width,
	 input_buffers.at(0).channels[0].height,
	 in0_padding_ch0);
      break;
     default:
       printf ("CUDA kernel Error: Unknown format \n");
       return false;
  }

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

bool cuda_mixer_ip (std::vector<GstCudaData> input_buffers, GstCudaData * output_buffer)
{
  cudaError_t cudaErr;
  cudaStream_t *stream;

  if (!check_inputs_resolution_match (input_buffers)) {
    printf ("Error: Mismatch between input buffers size\n");
    return false;
  }

  if (!check_resolution_match (input_buffers, output_buffer)) {
    printf ("Error: Mismatch between input and output buffer size\n");
    return false;
  }

  dim3 grid_dimen = dim3 (2, (input_buffers.at(0).channels[0].height / 2), 1);
  dim3 block_dimen = dim3 ((input_buffers.at(0).channels[0].width / 4), 1, 1);
  int in0_padding_ch0 = input_buffers.at(0).channels[0].pitch - input_buffers.at(0).channels[0].width;
  int in0_padding_ch1 = input_buffers.at(0).channels[1].pitch - input_buffers.at(0).channels[1].width;
  int in1_padding_ch0 = input_buffers.at(1).channels[0].pitch - input_buffers.at(1).channels[0].width;
  int in1_padding_ch1 = input_buffers.at(1).channels[1].pitch - input_buffers.at(1).channels[1].width;

  if (in0_padding_ch0 != in1_padding_ch0) {
    printf ("Error: Mismatch between inputs ch0 (Y) padding\n");
    return false;
  }

  if (input_buffers.at(0).format == GST_CUDA_I420 && in0_padding_ch1 != in1_padding_ch1) {
    printf ("Error: Mismatch between inputs ch1 (UV) padding\n");
    return false;
  }

  stream = (cudaStream_t*)output_buffer->stream;

  switch(input_buffers.at(0).format){
    case GST_CUDA_I420:
      cuda_accel_mixer_i420_ip  <<< grid_dimen, block_dimen, 0, *stream >>> (
	 (unsigned char *) input_buffers.at(1).channels[0].data,
	 (unsigned char *) input_buffers.at(1).channels[1].data,
	 (unsigned char *) input_buffers.at(1).channels[2].data,
	 (unsigned char *) output_buffer->channels[0].data,
	 (unsigned char *) output_buffer->channels[1].data,
	 (unsigned char *) output_buffer->channels[2].data,
	 input_buffers.at(0).channels[0].width,
	 input_buffers.at(0).channels[0].height,
	 in0_padding_ch0, in0_padding_ch1);
      break;
    case GST_CUDA_GREY:
      cuda_accel_mixer_grey8_ip  <<< grid_dimen, block_dimen, 0, *stream >>> (
	 (unsigned char *) input_buffers.at(1).channels[0].data,
	 (unsigned char *) output_buffer->channels[0].data,
	 input_buffers.at(0).channels[0].width,
	 input_buffers.at(0).channels[0].height,
	 in0_padding_ch0);
      break;
     default:
       printf ("CUDA kernel Error: Unknown format \n");
       return false;
  }

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

class Example : public Gst::Cuda::Algorithm::Mux
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

  bool process (std::vector<GstCudaData> input_buffers, GstCudaData &output_buffer) override {
    //Put here the process logic (This should be used when in_place is not configured)
    return cuda_mixer_not_ip (input_buffers, &output_buffer);
  }

  bool process_ip (std::vector<GstCudaData> input_buffers, GstCudaData &output_buffer) override {
    //Put here the process logic (This should be used when in_place is configured)
    return cuda_mixer_ip (input_buffers, &output_buffer);
  }
};

Gst::Cuda::Algorithm::Mux *
factory_make ()
{
  return new Example ();
}
