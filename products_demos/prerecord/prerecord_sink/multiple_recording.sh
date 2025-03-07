#!/bin/bash

# Pipeline definition

mkdir output

#Flag for setting up NVIDIA Jetson mode

jetson=false

while getopts j? flag
do
    case "${flag}" in
        j) jetson=true;;
    esac
done

if "$jetson"; then
    ENC="nvvidconv ! nvv4l2h264enc iframeinterval=15 idrinterval=15 insert-sps-pps=true maxperf-enable=true"
else
    ENC="x264enc speed-preset=ultrafast key-int-max=15"
fi

gstd-client pipeline_create p1 videotestsrc is-live=true name=src pattern=ball ! $ENC ! h264parse ! prerecordsink buffering=true on-key-frame=true name=prerecordsink location=output/recording_%Y-%m-%d_%H:%M:%S%z.mp4 buf-time=5000 max-buf-time=12000
# Run pipeline
gstd-client pipeline_play p1

# After 10 seconds, change the video pattern
sleep 10
gstd-client element_set p1 src pattern snow

# Allow buffer data to pass downstream 
gstd-client element_set p1 prerecordsink buffering false

sleep 11
gstd-client element_set p1 prerecordsink buffering false

# Stop the pipeline after 10 seconds
sleep 10
gstd-client pipeline_stop p1

# Delete pipeline
gstd-client pipeline_delete p1
