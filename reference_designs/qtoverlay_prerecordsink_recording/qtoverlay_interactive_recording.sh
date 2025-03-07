#!/bin/bash

# Copyright (C) 2024 RidgeRun, LLC (http://www.ridgerun.com)
# All Rights Reserved.
#
# The contents of this software are proprietary and confidential to RidgeRun,
# LLC.  No part of this program may be photocopied, reproduced or translated
# into another programming language without prior written consent of
# RidgeRun, LLC.  The user is free to modify the source code after obtaining
# a software license from RidgeRun.  All source code changes must be provided
# back to RidgeRun without any encumbrance.

echo -e "\n === GstPreRecordSink with interactive input ===\n"

END_MSJ="Finished stream"
record_time=5000
while getopts t:h? flag
do
    case "${flag}" in
        t) input_time=${OPTARG};;
	*|h)
	    echo "RidgeRun QtOverlay + PreRecord"
	    echo ""
	    echo "-t: Set recording time in miliseconds e.g -t 7000"
            exit 1
            ;;
    esac
done

mkdir output

[ ! -z "${num##*[!0-9]*}" ] && record_time=$input_time;

gstd-client pipeline_create p1 videotestsrc is-live=true name=src pattern=0 ! video/x-raw,width=1920,height=1080 ! qtoverlay name=ql qml=qt_prerecord.qml ! timeoverlay halignment=right valignment=top ! queue max-size-buffers=3 leaky=downstream ! tee name=t t. ! queue ! videoconvert ! x264enc speed-preset=ultrafast key-int-max=30 ! h264parse ! prerecordsink buffering=true on-key-frame=true name=prerecordsink location=output/recording_%Y_%m_%d_%H_%M_%S.mp4 buf-time=$record_time t. ! queue ! videoconvert ! autovideosink

# Function to change the video pattern
PATTERN=0
function change_src {
	echo $PATTERN
	if [ $(echo "$PATTERN < 25" | bc ) -eq 1 ];then
		gstd-client "element_set p1 src pattern $PATTERN"	
	else
		PATTERN=0
		gstd-client "element_set p1 src pattern $PATTERN" 
	fi
}

# Function that alloww buffer data to pass downstream 
function record {
	gstd-client element_set p1 ql qml-attribute "RecMain.visible:true" 
	gstd-client element_set p1 ql qml-attribute "GifMain.visible:true" 
	gstd-client element_set p1 ql qml-attribute "TextMain.visible:true" 
	gstd-client element_set p1 prerecordsink buffering false
	gstd-client element_set p1 ql qml-attribute "Main.text:Post-Trigger content" 
}

function stop_rec {
	gstd-client element_set p1 ql qml-attribute "RecMain.visible:false" 
	gstd-client element_set p1 ql qml-attribute "GifMain.visible:false" 
	gstd-client element_set p1 ql qml-attribute "TextMain.visible:false" 
	gstd-client element_set p1 ql qml-attribute "Main.text:Pre-Trigger content" 
}

# Run pipeline
gstd-client pipeline_play p1

echo "Running the pipeline"
active=true
while [ "$active" = true ];
do
	read -s -n 1 key
	case $key in
		 '')
			echo -e "\n====>Start/Extend recording for $record_time ms! \n"
			record
			PATTERN=$(("$PATTERN" + 1))
			change_src $PATTERN
			sleep $(($record_time / 1000))
			stop_rec
			;;
		$'\e')
			read -rsn2 -t 0.1 special
			case "$special" in
				'')
					echo "Escape"
					gstd-client pipeline_stop p1
					gstd-client pipeline_delete p1
					echo -e $END_MSJ
					break
					;;
			esac
	esac
done

