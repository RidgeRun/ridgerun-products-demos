#!/bin/bash

PS3='Input the number of the desired option: '
options=("Stream to file" "Change metadata string" "View file and extract meta" "Quit")
video_options=("H264" "H265" "Quit")
exit=0
METADATA='"Hello world"'

while getopts j? flag
do
    case "${flag}" in
        j) jetson=true;;
    esac
done

if "$jetson"; then
	ENC264="nvv4l2h264enc"
	DEC="nvv4l2decoder ! nvvidconv "
	ENC265="nvv4l2h265enc"
else
	ENC264="x264enc"
	DEC="decodebin "
	ENC265="x265enc"
fi

clear
# Menu to select example pipelines
while [ $exit -lt 1 ]
do 
    select opt in "${options[@]}"
    do
        # Create the recording file option
        case $opt in
            "Stream to file")
                clear
                PS3='Input number of the desired encoding type for video file: '
                # Pipelines stream 50 buffers to file. Stream stops after buffers are streamed
                echo -e "\nSelect a Video encoding format. File will be created after."
                select enc in "${video_options[@]}"
                do
                    case $enc in
                        "H264")
                            clear
                            echo -e $stop
                            gst-launch-1.0 videotestsrc num-buffers=50 ! $ENC264 ! seiinject metadata=$METADATA ! qtmux ! filesink location=h264_enc.mp4 -e
                            break
                            ;;
                        "H265")
                            clear
                            echo -e $stop
                            gst-launch-1.0 videotestsrc num-buffers=50 ! $ENC265 ! h265parse ! seiinject metadata=$METADATA ! qtmux ! filesink location=h265_enc.mp4 -e
                            break
                            ;;
                        "Quit")
                            break
                            ;;
                        *) echo "invalid option $REPLY";;
                    esac
                done
                PS3='Input the number of the desired option: '
                break
                ;;
            # Send any string in the metadata option
            "Change metadata string")
                echo -e "\nInput metadata string or press Enter to use default (Hello world): "
                read METADATA
                if [ -z "$METADATA" ]
                then
                    METADATA='"Hello world"'
                else 
                    METADATA=\'"$METADATA"\'
                fi 

                break
                ;;
            # View the recording file option
            "View file and extract meta")
                clear
                PS3='Input number of the desired encoding type video file: '
                return=0
                while [ $return -lt 1 ]
                do
                    # Pipelines will stop when the file is played back
                    echo -e "\nSelect a Video encoding format. File will play after"
                    select dec in "${video_options[@]}"
                    do
                        case $dec in
                            "H264")
                                clear
                                echo -e $stop
                                # DEBUG is enabled to view added metadata.
                                # Normally this output would be larger but we filter it to show the important meta only.
                                GST_DEBUG=*seiextract*:MEMDUMP gst-launch-1.0 filesrc location=h264_enc.mp4 ! queue ! qtdemux ! video/x-h264 ! \
                                seiextract ! h264parse ! $DEC !  queue ! videoconvert ! xvimagesink 2> >(grep -i -e MEMDUMP -e Error)
                                break
                                ;;
                            "H265")
                                clear
                                echo -e $stop
                                GST_DEBUG=*seiextract*:MEMDUMP gst-launch-1.0 filesrc location=h265_enc.mp4 ! queue ! qtdemux ! video/x-h265 ! \
                                seiextract !  \h265parse ! $DEC !  queue ! videoconvert ! xvimagesink 2> >(grep -i -e MEMDUMP -e Error)
                                break
                                ;;
                            "Quit")
                                return=1
                                break
                                ;;
                            *) echo "invalid option $REPLY";;
                        esac
                    done
                done
                PS3='Input the number of the desired option: '
                break
                ;;
            "Quit")
                exit=1
                break
                ;;
            *) echo "invalid option $REPLY";;
        esac
    done
done
