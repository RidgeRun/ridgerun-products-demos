#!/bin/bash

echo -e "\n === GstPreRecordSink with interactive input ===\n"

END_MSJ="Finished stream"

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

mkdir output

gstd-client pipeline_create p1 videotestsrc is-live=true name=src pattern=0 ! $ENC ! h264parse ! prerecordsink buffering=true on-key-frame=true name=prerecordsink location=output/recording_%H_%M_%S%z.mp4 buf-time=7000


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
function recording {
	gstd-client element_set p1 prerecordsink buffering false
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
			echo -e "\n====>Sending signal for recording! \n"
			recording
			;;
		'c')
			echo -e "\n====> Changing source! \n"
			PATTERN=$(("$PATTERN" + 1))
			change_src $PATTERN
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

