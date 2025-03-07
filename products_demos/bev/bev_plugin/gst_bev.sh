#!/bin/bash

CAPS="video/x-raw,format=RGBA"

#Flag for setting up NVIDIA Jetson mode

while getopts j? flag
do
    case "${flag}" in
        j) jetson=true;;
    esac
done

if "$jetson"; then
	ENC264=" nvvidconv ! nvv4l2h264enc iframeinterval=15 idrinterval=15 insert-sps-pps=true maxperf-enable=true"
else
	ENC264=" videoconvert ! x264enc key-int-max=15"
fi

GST_DEBUG=2 gst-launch-1.0 -e  bev name=bev0 calibration-file=birds_eye_view.json  \
	 filesrc location=../samples/bev_6_cameras/cam_0.jpg ! jpegparse ! jpegdec ! imagefreeze ! videoconvert ! $CAPS  !  queue ! bev0.sink_0 \
 	filesrc location=../samples/bev_6_cameras/cam_1.jpg ! jpegparse ! jpegdec ! imagefreeze ! videoconvert ! $CAPS ! queue ! bev0.sink_1 \
 	filesrc location=../samples/bev_6_cameras/cam_2.jpg ! jpegparse ! jpegdec ! imagefreeze ! videoconvert ! $CAPS ! queue ! bev0.sink_2 \
 	filesrc location=../samples/bev_6_cameras/cam_5.jpg ! jpegparse ! jpegdec ! imagefreeze ! videoconvert ! $CAPS ! queue ! bev0.sink_3 \
 	filesrc location=../samples/bev_6_cameras/cam_4.jpg ! jpegparse ! jpegdec ! imagefreeze ! videoconvert ! $CAPS ! queue ! bev0.sink_4 \
 	filesrc location=../samples/bev_6_cameras/cam_3.jpg ! jpegparse ! jpegdec ! imagefreeze ! videoconvert ! $CAPS ! queue ! bev0.sink_5 \
	bev0. ! queue ! $ENC264 ! h264parse ! matroskamux ! filesink  location=sample_bev.ts
