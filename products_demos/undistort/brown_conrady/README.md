# CUDA Undistort of Brown-Conrady distortion side by side

In this demo, you will see how CUDA Undistort is capable of applying Brown-Conrady distortion model to a distorted video file. The demo uses the display to show both, distorted and undistorted videos, side by side in a video composition.

This example uses the following RidgeRun products:
* [Cuda Undistort's wiki](https://developer.ridgerun.com/wiki/index.php/CUDA_Accelerated_GStreamer_Camera_Undistort)

# How to run it

1) Download the samples from the sample folder

> ../samples/download_sample.sh

2) Execute the script

> ./undistorted_vs_distored.sh

# Troubleshooting

The first level of debug to troubleshoot a failing evaluation binary is to inspect GStreamer debug output.

> GST_DEBUG=2 ./undistorted_vs_distored.sh

If the output doesn't help you figure out the problem, please contact support@ridgerun.com with the output of the GStreamer debug and any additional information you consider useful.

