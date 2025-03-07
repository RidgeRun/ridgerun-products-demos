# Birds Eye View GstPlugin example

In this demo, you will see how you can generate an aerial view from 6 inputs by using the Birds Eye View plugin.   

This example uses the following RidgeRun products:
* [Birds Eye View's wiki](https://developer.ridgerun.com/wiki/index.php/Birds_Eye_View)

# How to run it

1) Download the samples from the sample folder

> ../samples/download_sample.sh

2) Execute the script

> ./gst_bev.sh

3) Once you finished recording with '''Ctrl+C'''. You can watch the video result.

> vlc sample_bev.ts

## NVIDIA Jetson Platform

For executing the demo in a NVIDIA Jetson Platform use the following flag:

```
./gst_bev.sh -j 
```

# Troubleshooting

The first level of debug to troubleshoot a failing evaluation binary is to inspect GStreamer debug output. 

GST_DEBUG=2 ./gst_bev.sh

If the output doesn't help you figure out the problem, please contact support@ridgerun.com with the output of the GStreamer debug and any additional information you consider useful.

