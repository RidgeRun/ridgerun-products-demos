#!/bin/bash

source ENV_VARIABLES 

INPUT_0="../samples/Park_30s/360-s0.mp4"
INPUT_1="../samples/Park_30s/360-s1.mp4"

OUTPUT=360_stitching_result.mp4

#Flag for setting up NVIDIA Jetson mode

jetson=false

while getopts j? flag
do
    case "${flag}" in
        j) jetson=true;;
    esac
done

if "$jetson"; then
	DEC="nvv4l2decoder ! nvvidconv"
    ENC="nvvidconv ! nvv4l2h264enc bitrate=30000000"
else
    DEC="decodebin ! videoconvert"
	ENC="videoconvert ! x264enc"
fi

gst-launch-1.0  cudastitcher name=stitcher \
	homography-list="`cat result.json | tr -d "\n" | tr -d "\t" | tr -d " "`" \
	filesrc location="$INPUT_0" ! qtdemux ! queue ! h264parse ! $DEC ! rrfisheyetoeqr radius=$R0 lens=$L0 center-x=$CX0 center-y=$CY0 rot-x=$RX0 rot-y=$RY0 rot-z=$RZ0 name=proj0 !  queue ! stitcher.sink_0 \
	filesrc location="$INPUT_1" ! qtdemux ! queue ! h264parse ! $DEC ! rrfisheyetoeqr radius=$R1 lens=$L1 center-x=$CX1 center-y=$CY1 rot-x=$RX1 rot-y=$RY1 rot-z=$RZ1 name=proj1 !  queue ! stitcher.sink_1 \
	stitcher. ! queue ! $ENC ! h264parse ! queue !  qtmux ! filesink location=$OUTPUT -e
