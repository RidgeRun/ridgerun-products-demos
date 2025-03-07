#!/bin/bash 

#Flag for setting up NVIDIA Jetson mode

jetson=false

while getopts j? flag
do
    case "${flag}" in
        j) jetson=true;;
    esac
done

if "$jetson"; then
    CAPS="video/x-raw(memory:NVMM), width=1920, height=1080"
	DEC="nvv4l2decoder ! nvvidconv ! $CAPS, format=RGBA"
    ENC="nvvidconv ! $CAPS ! nvv4l2h264enc bitrate=30000000"
else
    CAPS="video/x-raw, width=1920, height=1080, format=RGBA"
    DEC="decodebin ! videoconvert ! $CAPS"
	ENC="videoconvert ! x264enc"
fi

INPUT_0=../samples/Mosaic_1min/mosaic_static_vid1_1m_s4.mp4 
INPUT_1=../samples/Mosaic_1min/mosaic_static_vid1_1m_s3.mp4 
INPUT_2=../samples/Mosaic_1min/mosaic_static_vid1_1m_s0.mp4 
INPUT_3=../samples/Mosaic_1min/mosaic_static_vid1_1m_s1.mp4 
INPUT_4=../samples/Mosaic_1min/mosaic_static_vid1_1m_s2.mp4 
INPUT_5=../samples/Mosaic_1min/mosaic_static_vid1_1m_s5.mp4 

OUTPUT=stitching_result.mp4

gst-launch-1.0 -e cudastitcher name=stitcher homography-list="`cat homography.json | tr -d "\n" | tr -d " "`" filesrc location=$INPUT_0 ! qtdemux ! h264parse ! $DEC ! queue ! stitcher.sink_0 \
    filesrc location=$INPUT_1 ! qtdemux ! h264parse ! $DEC ! queue ! stitcher.sink_1 \
    filesrc location=$INPUT_2 ! qtdemux ! h264parse ! $DEC ! queue ! stitcher.sink_2 \
    filesrc location=$INPUT_3 ! qtdemux ! h264parse ! $DEC ! queue ! stitcher.sink_3 \
    filesrc location=$INPUT_4 ! qtdemux ! h264parse ! $DEC ! queue ! stitcher.sink_4 \
    filesrc location=$INPUT_5 ! qtdemux ! h264parse ! $DEC ! queue ! stitcher.sink_5 \
    stitcher. ! queue ! $ENC ! h264parse ! mp4mux ! filesink location=$OUTPUT
