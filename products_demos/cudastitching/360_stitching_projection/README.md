# 360_stitching_projection

This demo showcases how cuda stitcher alongside the GstProjector converts the input from two fish eyes cameras in one single stream of 360 field of view.

Each stream will last for 1 minute and finally generates a 1 video stream of 360 degrees.

This example uses the following RidgeRun products:
* [GstCUDA developer's wiki](https://developer.ridgerun.com/wiki/index.php/GstCUDA)
* [GstStitcher developer's wiki](https://developer.ridgerun.com/wiki/index.php/Image_Stitching_for_NVIDIA_Jetson)
* [GstProjector developer's wiki](https://developer.ridgerun.com/wiki/index.php/RidgeRun_Image_Projector)

# How to run it

1) Donwload the samples from the sample folder directory. 

2) Execute the script.

> ./360_stitching.sh

## NVIDIA Jetson Platform

For executing the demo in a NVIDIA Jetson Platform use the following flag:

```
./360_stitching.sh -j 
```

# Troubleshooting

The first level of debug to troubleshoot a failing evaluation binary is to inspect GStreamer debug output. 

GST_DEBUG=2 ./360_stitching.sh

If the output doesn't help you figure out the problem, please contact support@ridgerun.com with the output of the GStreamer debug and any additional information you consider useful.
