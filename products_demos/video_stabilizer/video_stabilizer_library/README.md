# Video Stabilizer Library Implementation

This demos showcases how Video Stablizer can stablize the video from a prerecorded file using gyro data based on a IMU as a input.

The video generated can be reproduced to see the results.

This example uses the following RidgeRun product: 
* [Video Stabilizer's wiki]()

# How you can compile it.

1) Make sure that you have already installed Video Stablizer. 

2) Download before hand the samples for the compiliation in the samples folder.

3) Then you can proceed with the compilation as follows.

> make

This will generate a executable.

rvs-complete-concept

# How to run it

Test the example with the following line:

```bash
WIDTH=1280
HEIGHT=720
BACKEND=opencv
FOV=2.4
./rvs-complete-concept -f ../samples/non_stabilized.mp4 -g ../samples/raw-gyro-data.csv -b $BACKEND -w $WIDTH -h $HEIGHT -s $FOV
```

While increasing the width and height, you may need to decrease the `FOV`.

The expected output would be a video named output.mp4

# Troubleshooting

The first level of debug to troubleshoot a failing evaluation binary is to inspect GStreamer debug output.

> GST_DEBUG=2 ./rvs-complete-concept -f ../samples/non_stabilized.mp4 -g ../samples/raw-gyro-data.csv -b $BACKEND -w $WIDTH -h $HEIGHT -s $FOV

If the output doesn't help you figure out the problem, please contact support@ridgerun.com with the output of the GStreamer debug and any additional information you consider useful.

