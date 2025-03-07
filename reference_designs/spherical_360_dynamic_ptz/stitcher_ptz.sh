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


echo -e "\n === Stitcher + spherical PTZ with Display ===\n"

END_MSJ="Finished stream"

trap "gstd-client pipeline_delete p1; echo -e $END_MSJ; exit" SIGHUP SIGINT SIGTERM

#Defining variables for the demo

file="homographies.json"
HOMOGRAPHIES="`cat $file | tr -d "\n" | tr -d " "`"

# Initialize arrays
RADIUS=()
LENS=()
CENTERX=()
CENTERY=()
ROTX=()
ROTY=()
ROTZ=()

# Loop through projections
for ((i=0; i<3; i++)); do
  echo "Projection $i:"
  radius=$(echo "$HOMOGRAPHIES" | grep -oP "\"radius\":\s*\K[0-9]+(?=\.)" | sed -n "$((i+1))p")
  lens=$(echo "$HOMOGRAPHIES" | grep -oP "\"lens\":\s*\K[0-9.]+(?=,)" | sed -n "$((i+1))p")
  center_x=$(echo "$HOMOGRAPHIES" | grep -oP "\"center_x\":\s*\K[0-9]+(?=\.)" | sed -n "$((i+1))p")
  center_y=$(echo "$HOMOGRAPHIES" | grep -oP "\"center_y\":\s*\K[0-9]+(?=\.)" | sed -n "$((i+1))p")
  rot_x=$(echo "$HOMOGRAPHIES" | grep -oP "\"rot_x\":\s*\K[-0-9.]+" | sed -n "$((i+1))p")
  rot_y=$(echo "$HOMOGRAPHIES" | grep -oP "\"rot_y\":\s*\K[-0-9.]+" | sed -n "$((i+1))p")
  rot_z=$(echo "$HOMOGRAPHIES" | grep -oP "\"rot_z\":\s*\K[-0-9.]+" | sed -n "$((i+1))p")
  # Append values to arrays
  RADIUS+=($radius)
  LENS+=($lens)
  CENTERX+=($center_x)
  CENTERY+=($center_y)
  ROTX+=($rot_x)
  ROTY+=($rot_y)
  ROTZ+=($rot_z)
done

PAN=0
TILT=0
ZOOM=1
automode_flag=false
mp4_flag=false
no_display_flag=false
rtsp_flag=false
PORT=()
TARGET_TILT=0
TARGET_PAN=0
TARGET_ZOOM=1.0

#Flag for setting up NVIDIA Jetson mode

jetson=false

# Parsing flags for the demo
while getopts "a?m:c:n?r:j?h?" flag; do
    case "${flag}" in
	a)
	    automode_flag=true
	    ;;
    m)
        mp4_flag=true
	    IFS=' ' read -r -a mp4_files <<< "$OPTARG"
        ;;
	n)
	    no_display_flag=true
	    ;;
	r)
            rtsp_flag=true
	    PORT+=("$OPTARG")
	    ;;
    j) jetson=true
		;;
	*|h)
	    echo "RidgeRun Stitcher + PTZ + QTOverlay Demo"
	    echo ""
	    echo '-m : Load mp4 files e.g -m "1.mp4 2.mp4 3.mp4"'
	    echo "-n : Disable display"
	    echo "-r: Enables RTSP output with the specific port e.g -r 7000"
	    echo "-j: Enable NVIDIA Jetson mode"
	    echo "-a: Automatic PTZ mode"
            exit 1
            ;;
    esac
done

if "$jetson"; then
    ENC="nvvidconv ! nvv4l2h264enc bitrate=60000000 idrinterval=10 iframeinterval=10 insert-aud=true insert-vui=true qos=true insert-sps-pps=true profile=4 maxperf-enable=true"
	DISP="nvvidconv ! queue leaky=downstream ! nveglglessink sync=true"
	DEC="nvv4l2decoder ! nvvidconv"
	INF="nvvidconv ! mux.sink_0 nvstreammux name=mux batch-size=1 batched-push-timeout=40000 width=1080 height=1080 live-source=false ! queue leaky=2 max-size-buffers=3 ! perf name=before_infer ! nvinfer name=nvinfer1 config-file-path="${HOME}"/360-inference/peoplenet/config_infer_primary_peoplenet.txt ! perf name=after_infer  ! queue ! nvvidconv ! nvdsosd"
else
	ENC="videoconvert ! x264enc speed-preset=ultrafast key-int-max=30"
	DISP="videoconvert ! autovideosink async=true"
	DEC="avdec_h264 ! videoconvert"
	INF="identity"
fi

shift $((OPTIND -1))

#Checking flags cases for outputs
if [ "$no_display_flag" = true -a "$rtsp_flag" = false ];then
	echo "You need to have at least one output between rtsp and the display" >&2
	exit 1
fi


#Create pipeline for rtsp case
if [ "$rtsp_flag" = true ]; then
	echo "Using port: ${PORT}"
	gstd-client pipeline_create p3 interpipesrc name=rtsp listen-to=input_src allow-renegotiation=false format=time ! queue max-size-buffers=5 leaky=2 ! $ENC ! h264parse ! video/x-h264, stream-format=avc, mapping=stream1 ! rtspsink service=$PORT async-handling=true
fi
#Create pipeline for display case
if [ "$no_display_flag" = false ];then 
	gstd-client pipeline_create p2 interpipesrc name=output listen-to=input_src format=time ! queue leaky=2 max-size-buffers=5 ! $DISP
fi

#Create mp4 pipelines
if [ "$mp4_flag" = true ]; then
    if [[ "${mp4_files[0]}" == *.mp4 ]] && [[ "${mp4_files[1]}" == *.mp4 ]]  && [[ "${mp4_files[2]}" == *.mp4 ]]; then
		MP41="${mp4_files[0]}"
		MP42="${mp4_files[1]}"
		MP43="${mp4_files[2]}"
    else
        echo "Error: not valid .mp4 file."
        exit 1
    fi
else
	MP41=../samples/Example1/cam1.mp4
	MP42=../samples/Example1/cam2.mp4
	MP43=../samples/Example1/cam3.mp4
fi

gstd-client pipeline_create p1 cudastitcher name=stitcher homography-list=${HOMOGRAPHIES} sync=true \
filesrc location="${MP41}" ! qtdemux ! h264parse ! $DEC ! queue ! rrfisheyetoeqr crop=false center_x="${CENTERX[0]}" center_y="${CENTERY[0]}" radius="${RADIUS[0]}"  rot-x="${ROTX[0]}"  rot-y="${ROTY[0]}"  rot-z="${ROTZ[0]}"  lens="${LENS[0]}"  name=proj0 !  queue ! stitcher.sink_0 \
filesrc location="${MP42}" ! qtdemux ! h264parse ! $DEC ! queue ! rrfisheyetoeqr crop=false center_x="${CENTERX[1]}" center_y="${CENTERY[1]}" radius="${RADIUS[1]}"  rot-x="${ROTX[1]}"  rot-y="${ROTY[1]}"  rot-z="${ROTZ[1]}"  lens="${LENS[1]}"  name=proj1 !  queue ! stitcher.sink_1 \
filesrc location="${MP43}" ! qtdemux ! h264parse ! $DEC ! queue ! rrfisheyetoeqr crop=false center_x="${CENTERX[2]}" center_y="${CENTERY[2]}" radius="${RADIUS[2]}"  rot-x="${ROTX[2]}"  rot-y="${ROTY[2]}"  rot-z="${ROTZ[2]}"  lens="${LENS[2]}"  name=proj2 !  queue ! stitcher.sink_2 \
stitcher.  ! queue ! rrpanoramaptz name=ptz  ! qtoverlay qml=logos.qml ! queue leaky=2 max-size-buffers=3 ! $INF ! queue leaky=2 max-size-buffers=5 ! interpipesink name=input_src sync=true async=false forward-eos=true

#Function to change the pan
function change_pan {
	if [ "$1" -le -359 ] || [ "$1" -ge 360 ]; 
	then
		PAN=0
	fi
	gstd-client "element_set p1 ptz pan $1"
}

#Function to change the tilt
function change_tilt {
	if [ "$1" -le -360 ] || [ "$1" -ge 360 ]; 
	then
		TILT=0
	fi
	gstd-client "element_set p1 ptz tilt $1"
}

#Function to change the zoom
function change_zoom {
	_zoom=$1
	_low=0.2
	_high=9.9
	checking_zoom=$(echo "${_zoom}<${_low}" | bc)
	if [[ $checking_zoom = 1 ]]; 
	then
		ZOOM=0.2
		_zoom=$ZOOM
	fi
	checking_zoom=$(echo "${_zoom}>${_high}" | bc)
	if [[ $checking_zoom = 1 ]]; 
	then
		ZOOM=9.9
		_zoom=$ZOOM
	fi
	gstd-client "element_set p1 ptz zoom $_zoom"
}

#Function to set the automatic mode
function automatic_mode {
	n=1
	while [ "$n" -le 25 ]; 
	do
		read -t 0.1 -n 1 key
        	if [[ "$key" == "0" ]]; then
			automode_flag=false
			return 0
		fi
		ZOOM=$(echo "$ZOOM + 0.1" | bc )
		change_zoom $ZOOM
		sleep 0.02
		((n++))
	done
	n=1
	while [ "$n" -le 20 ];
	do
		read -t 0.1 -n 1 key
        	if [[ "$key" == "0" ]]; then
			automode_flag=false
			return 0
		fi
		((TILT--))
		change_tilt $TILT
		sleep 0.02
		((n++))
	done
	n=1
	while [ "$n" -le 20 ];
	do
		read -t 0.1 -n 1 key
        	if [[ "$key" == "0" ]]; then
			automode_flag=false
			return 0
		fi
		((TILT++))
		change_tilt $TILT
		sleep 0.02
		((n++))
	done
	n=1
	while [ "$n" -le 28 ];
	do
		read -t 0.1 -n 1 key
        	if [[ "$key" == "0" ]]; then
			automode_flag=false
			return 0
		fi
		ZOOM=$(echo "$ZOOM - 0.1" | bc )
		change_zoom $ZOOM
		sleep 0.02
		((n++))
	done
	n=1
	while [ "$n" -le 3 ]; 
	do
		read -t 0.1 -n 1 key
        	if [[ "$key" == "0" ]]; then
			automode_flag=false
			return 0
		fi
		ZOOM=$(echo "$ZOOM + 0.1" | bc )
		change_zoom $ZOOM
		sleep 0.02
		((n++))
	done
	n=1
	while [ "$n" -le 20 ];
	do
		read -t 0.1 -n 1 key
        	if [[ "$key" == "0" ]]; then
			automode_flag=false
			return 0
		fi
		((TILT++))
		change_tilt $TILT
		sleep 0.02
		((n++))
	done
	n=1
	while [ "$n" -le 20 ];
	do
		read -t 0.1 -n 1 key
        	if [[ "$key" == "0" ]]; then
			automode_flag=false
			return 0
		fi
		((TILT--))
		change_tilt $TILT
		sleep 0.02
		((n++))
	done
	n=1
	while [ "$n" -le 90 ];
	do
		read -t 0.1 -n 1 key
        	if [[ "$key" == "0" ]]; then
			automode_flag=false
			return 0
		fi
		((PAN--))
		change_pan $PAN
		sleep 0.02
		((n++))
	done
	n=1
	while [ "$n" -le 180 ];
	do
		read -t 0.1 -n 1 key
        	if [[ "$key" == "0" ]]; then
			automode_flag=false
			return 0
		fi
		((PAN++))
		change_pan $PAN
		sleep 0.02
		((n++))
	done
	n=1
	while [ "$n" -le 90 ];
	do
		read -t 0.1 -n 1 key
        	if [[ "$key" == "0" ]]; then
			automode_flag=false
			return 0
		fi
		((PAN--))
		change_pan $PAN
		sleep 0.02
		((n++))
	done
}

#Function for extracting the type of step for the reset position
function get_step {
	local last_value="$1"
	local target="$2"
	if [  $(echo "$target > 180" | bc) -eq 1 ];then
                if [  $(echo "$last_value > 0" | bc) -eq 1 ];then
                        last_value=$(("$last_value"-360))
                fi
                target=$(("$target"-360))
                if [  $(echo "$last_value < $target" | bc) -eq 1  -a  $(echo "$last_value > ($target-180)" | bc) -eq 1 ];then
                        step=1
                else
                        step=-1
                fi
        elif [  $(echo "$target == 0" | bc) -eq 1 ];then
                if [  $(echo "$last_value < 0" | bc) -eq 1 ];then
                        last_value=$(("$last_value"+360))
                fi
                if [  $(echo "$last_value < $target+180" | bc) -eq 1 ];then
                        step=-1
                else
                        step=1
                fi
        else
                if [  $(echo "$last_value < 0" | bc) -eq 1 ];then
                        last_value=$(("$last_value"+360))
                fi
                if [  $(echo "$last_value > $target" | bc) -eq 1  -a  $(echo "$last_value < ($target+180)" | bc) -eq 1 ];then
                        step=-1
                else
                        step=1
                fi

        fi
	echo "$step"
	echo "$last_value"
	echo "$target"
}

#Function for reseting to the initial position
function reseting_position {
	local last_zoom="$ZOOM"
	zoom_step=0.1
	checking_zoom=$(echo "${last_zoom}>${TARGET_ZOOM}" | bc)
	if [ $checking_zoom = 1 ];then
		zoom_step=-0.1
	fi
	read -r -d ' ' pan_step new_pan target_pan <<< "$( get_step $PAN $TARGET_PAN )"
	PAN=$new_pan
        while [ "$PAN" != "$target_pan" ];
        do
                PAN=$(("$PAN"+"$pan_step"))
		change_pan $PAN
        done
	read -r -d ' ' tilt_step new_tilt target_tilt <<< "$( get_step $TILT $TARGET_TILT )"
	TILT=$new_tilt
        while [ "$TILT" != "$target_tilt" ];
        do
                TILT=$(("$TILT"+"$tilt_step"))
		change_tilt $TILT
        done

	seqzoom=$last_zoom
	while (( $(echo "$seqzoom != $TARGET_ZOOM" | bc -l) )); do
		change_zoom $seqzoom
		seqzoom=$(echo "$seqzoom + $zoom_step" | bc -l)
    		if (( $(echo "$zoom_step > 0 && $seqzoom > $TARGET_ZOOM" | bc -l) )) || \
			(( $(echo "$zoom_step < 0 && $seqzoom < $TARGET_ZOOM" | bc -l) )); then
        		break
    		fi
	done
	ZOOM="$TARGET_ZOOM"
	change_zoom $ZOOM


}

#Function for setting new reset position
function set_new_reset {
	TARGET_PAN="$PAN"
	TARGET_TILT="$TILT"
	TARGET_ZOOM="$ZOOM"
	if [  $(echo "$TARGET_PAN < 0" | bc) -eq 1 ];then
		TARGET_PAN=$(("$TARGET_PAN"+360))
	fi
	if [  $(echo "$TARGET_TILT < 0" | bc) -eq 1 ];then
		TARGET_TILT=$(("$TARGET_TILT"+360))
	fi

}

#Function for starting the automatic mode in loop
function initiating_loop {
	while [[ "$automode_flag" = true ]]; do
		echo "\n====> Activating automatic mode"
		automatic_mode
	done
}


#Set the pipelines in play status
gstd-client pipeline_play p1
if [ "$no_display_flag" = false ];then
	gstd-client pipeline_play p2
fi
if [ "$rtsp_flag" = true ];then
	gstd-client pipeline_play p3
fi
active=true

#Loop for receiving the controls for the demo
while [ "$active" = true ];
do  
	initiating_loop
	read -s -n 1 key
	case $key in 
		'+')
			echo -e "\n====> Zooming in! \n"
			ZOOM=$(echo "$ZOOM + 0.1" | bc )
			change_zoom $ZOOM
			;; 
		'-')
			echo -e "\n====> Zooming out! \n"
			ZOOM=$(echo "$ZOOM - 0.1" | bc )
			change_zoom $ZOOM
			;;
		'0')
			echo "\n====> Activating automatic mode"
                        automode_flag=true
			reseting_position
			initiating_loop
                        ;;
		'1')
			echo "\n====> Reseting position"
			reseting_position
                        ;;
		'2')
			echo "\n====> Setting new reset position"
			set_new_reset
                        ;;
		$'\e')
			read -rsn2 -t 0.1 special
			case "$special" in
				'[A') 
					echo "\n====> Turning Up \n" 
					((TILT--))
					change_tilt $TILT
					;;
				'[B') 
					echo -e "\n====> Turning Down! \n"
					((TILT++))
					change_tilt $TILT
					;; 
				'[C') 
					echo "\n====> Turning Right \n"
					((PAN++))
					change_pan $PAN
					;;
				'[D') 
					echo "\n====> Turning Left \n"
					((PAN--))
					change_pan $PAN
					;;
				'') 
					echo "Escape"
					gstd-client pipeline_stop p1
					gstd-client pipeline_delete p1
					if [ "$no_display_flag" = false ];then
						gstd-client pipeline_stop p2
						gstd-client pipeline_delete p2
					fi
					if [ "$rtsp_flag" = true ];then
						gstd-client pipeline_stop p3
						gstd-client pipeline_delete p3
					fi
					echo -e $END_MSJ
					break
					;;
			esac
	esac
done

