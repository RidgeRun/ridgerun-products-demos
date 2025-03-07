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
#include <sys/cuda/filteralgorithm.hpp>
#define __noinline__ __attribute__((noinline))

__global__ void
cuda_accel_grayscale_not_ip (unsigned char *InY, unsigned char *InU,
    unsigned char *InV, unsigned char *OutY,
    unsigned char *OutU, unsigned char *OutV,
    int width, int height, int stride_y, int stride_uv)
{
  /* Compute indexes */
  int yIndex = (2 * width * blockIdx.y) + (2 * threadIdx.x) +
      ((width / 2) * (blockIdx.x));

  int yIndexPitchOdd = yIndex + (2 * stride_y * blockIdx.y);
  int yIndexpitchEven = yIndexPitchOdd + stride_y;

  int uvIndex = ((width / 2) * blockIdx.y) +
      threadIdx.x + ((width / 4) * blockIdx.x);

  OutU[uvIndex] = 128;
  OutV[uvIndex] = 128;

  OutY[yIndex] = InY[yIndexPitchOdd];
  OutY[yIndex + 1] = InY[yIndexPitchOdd + 1];
  OutY[yIndex + width] = InY[yIndexpitchEven + width];
  OutY[yIndex + width + 1] = InY[yIndexpitchEven + width + 1];
}

__global__ void
cuda_accel_grayscale_ip (unsigned char *IO_U,
    unsigned char *IO_V, int width, int stride_uv)
{
  /* Compute indexes */
  int uvIndex = ((width / 2) * blockIdx.y) +
      threadIdx.x + ((width / 4) * blockIdx.x);

  int uvIndexPitch = uvIndex + ((stride_uv) * blockIdx.y);

  IO_U[uvIndexPitch] = 128;
  IO_V[uvIndexPitch] = 128;
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

bool cuda_grayscale_not_ip (const GstCudaData * input_buffer,
		GstCudaData * output_buffer)
{
  cudaError_t cudaErr;
  cudaStream_t *stream;

  if (!check_resolution_match (input_buffer, output_buffer)) {
    printf ("Error: Mismatch between input and output buffer size\n");
    return false;
  }

  dim3 grid_dimen = dim3 (2, (input_buffer->channels[0].height / 2), 1);
  dim3 block_dimen = dim3 ((input_buffer->channels[0].width / 4), 1, 1);
  int padding_ch0 = input_buffer->channels[0].pitch - input_buffer->channels[0].width;
  int padding_ch1 = input_buffer->channels[1].pitch - input_buffer->channels[1].width;

  stream = (cudaStream_t*)input_buffer->stream;

  cuda_accel_grayscale_not_ip  <<< grid_dimen, block_dimen, 0, *stream >>> ((unsigned char *) input_buffer->channels[0].data,
    (unsigned char *) input_buffer->channels[1].data, (unsigned char *) input_buffer->channels[2].data,
    (unsigned char *) output_buffer->channels[0].data, (unsigned char *) output_buffer->channels[1].data,
    (unsigned char *) output_buffer->channels[2].data, input_buffer->channels[0].width,
    input_buffer->channels[0].height, padding_ch0, padding_ch1);

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

bool cuda_grayscale_ip (GstCudaData * io_buffer)
{
  cudaError_t cudaErr;
  cudaStream_t *stream;

  dim3 grid_dimen = dim3 (2, (io_buffer->channels[0].height / 2), 1);
  dim3 block_dimen = dim3 ((io_buffer->channels[0].width / 4), 1, 1);
  int padding_ch1 = io_buffer->channels[1].pitch - io_buffer->channels[1].width;

  stream = (cudaStream_t*)io_buffer->stream;

  cuda_accel_grayscale_ip  <<< grid_dimen, block_dimen, 0, *stream >>> ((unsigned char *) io_buffer->channels[1].data,
    (unsigned char *) io_buffer->channels[2].data,
    io_buffer->channels[0].width, padding_ch1);

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
    return cuda_grayscale_not_ip (&input_buffer, &output_buffer);
  }

  bool process_ip (GstCudaData &io_buffer) override {
    //Put here the process logic (This should be used when in_place is configured)
    return cuda_grayscale_ip (&io_buffer);
  }
};

Gst::Cuda::Algorithm::Filter *
factory_make ()
{
  return new Example ();
}
