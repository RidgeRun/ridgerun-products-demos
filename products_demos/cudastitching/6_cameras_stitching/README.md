# 6_cameras_stitching

This demo showcases how cuda stitcher takes the input from 6 cameras and generate one stream. 

Each video with a lenght of 1 minute will be stitched to one single stream.

This example uses the following RidgeRun products:
* [GstCUDA developer's wiki](https://developer.ridgerun.com/wiki/index.php/GstCUDA)
* [GstStitcher developer's wiki](https://developer.ridgerun.com/wiki/index.php/Image_Stitching_for_NVIDIA_Jetson)

# How to run it

1) Donwload the samples from the sample folder directory. 

2) Execute the script.

> ./video_stitching.sh

## NVIDIA Jetson Platform

For executing the demo in a NVIDIA Jetson Platform use the following flag:

```
./video_stitching.sh -j 
```

# Troubleshooting

The first level of debug to troubleshoot a failing evaluation binary is to inspect GStreamer debug output. 

GST_DEBUG=2 ./video_stitching.sh

If the output doesn't help you figure out the problem, please contact support@ridgerun.com with the output of the GStreamer debug and any additional information you consider useful.

