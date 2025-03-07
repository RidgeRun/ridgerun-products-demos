#!/bin/bash

# Copyright (C) 2025 RidgeRun, LLC (http://www.ridgerun.com)
# All Rights Reserved.
#
# The contents of this software are proprietary and confidential to RidgeRun,
# LLC.  No part of this program may be photocopied, reproduced or translated
# into another programming language without prior written consent of
# RidgeRun, LLC.  The user is free to modify the source code after obtaining
# a software license from RidgeRun.  All source code changes must be provided
# back to RidgeRun without any encumbrance.

echo -e "\n === GstCUDA Gray Scale Filter ===\n"

camera_flag=false
CAMERA_ID=()
mp4_flag=false
display_flag=true
OUTVIDEO=gray_scale_filter_test.mp4

while getopts "m:w:h:d?help?" flag; do
    case "${flag}" in
    m)
        mp4_flag=true
	IFS=' ' read -r -a mp4_file <<< "$OPTARG"
        ;;
   w)
	WD="$OPTARG"
	;;
   h)
	HD="$OPTARG"
	;;
   d)
	display_flag=true
	;;
   *|help)
        echo "RidgeRun GstCUDA Gray Scale FIlter"
	echo ""
	echo '-m : Load mp4 files e.g -m "1.mp4"'
	echo "-d : Enable display (default record a video)"
	echo "-w: Define Width (default 1920) e.g -w 3820"
	echo "-h: Define Height (default 1080) e.g -w 650"
	exit 1
	;;
    esac
done

shift $((OPTIND -1))
W="${WD:-1920}"
H="${HD:-1080}"


if [ "$mp4_flag" = true ]; then
    if [[ "${mp4_file}" == *.mp4 ]]; then
		INPUT="filesrc location="${mp4_file}" ! decodebin ! nvvidconv ! video/x-raw,width="$W",height="$H",format=I420"
    else
        echo "Error: not valid .mp4 file."
        exit 1
    fi
else
	INPUT="videotestsrc is-live=true ! video/x-raw,width="$W",height="$H",format=I420,framerate=60/1 "
fi

echo $INPUT

#Create pipeline for display case
if [ "$display_flag" = false ];then
	OUTPUT="nveglglessink sync=true"
else
	OUTPUT="nvvidconv ! nvv4l2h264enc bitrate=20000000 ! h264parse ! mp4mux ! filesink location=$OUTVIDEO"
fi

GST_DEBUG=2 gst-launch-1.0 $INPUT ! nvvidconv ! "video/x-raw(memory:NVMM),width=$W,height=$H,format=I420" ! \
cudafilter in-place=true location=./gray-scale-filter.so ! $OUTPUT


