#!/bin/bash

PS3='Select the media type to stream: '
stop="\n To stop stream press Ctrl + C \n"
options=("Video" "Audio" "Stream from file" "Change PORT/MAPPING" "Quit")
video_options=("MPEG4" "H264" "H265" "VP8" "VP9" "Motion JPEG" "Quit")
audio_options=("AAC" "AC3" "PCMU" "PCMA" "OPUS" "Quit")
file_options=("Start stream from file" "Set file path" "Quit")
exit=0

# Buffers to create 2min mp4 recording
VID_BUFFERS=2000
AUDIO_BUFFERS=2750

dir=$PWD/../../resources/
RESOURCES_DIR=$(builtin cd $dir; pwd)

PORT=5000
MAPPING=/stream1
FILE=$RESOURCES_DIR/test_audio_video.mp4

while getopts j? flag
do
    case "${flag}" in
        j) jetson=true;;
    esac
done

if "$jetson"; then
	ENC264=" nvvidconv ! nvv4l2h264enc iframeinterval=15 idrinterval=15 insert-sps-pps=true maxperf-enable=true"
	ENC265=" nvvidconv ! nvv4l2h265enc iframeinterval=15 idrinterval=15 insert-sps-pps=true maxperf-enable=true"
	JPEGENC=" nvvidconv ! nvjpegenc "
else
	ENC264="x264enc key-int-max=15"
	ENC265='x265enc option-string="keyint=30:min-keyint=30:repeat-headers=1"'
	JPEGENC="jpegenc"
fi

clear
# Menu to select example pipelines to stream
while [ $exit -lt 1 ]
do 
    select opt in "${options[@]}"
    do
        # Video options
        case $opt in
            "Video")
                clear
                PS3='Select the encoding type for video stream: '
                return=0
                while [ $return -lt 1 ]
                do
                    # Pipelines have no streaming limit. Allow user to stop manually
                    echo -e "\nSelect a Video encoding format. Stream will start automatically." 
                    echo -e "Launch the rtsp_client.sh in a new terminal after the stream has started. \n"
                    select enc in "${video_options[@]}"
                    do
                        case $enc in
                            "MPEG4")
                                clear
                                echo -e $stop
                                VENC='avenc_mpeg4 ! video/mpeg'
                                echo $VENC
                                break
                                ;;
                            "H264")
                                clear
                                echo -e $stop
                                VENC="$ENC264 ! video/x-h264"
                                break
                                ;;
                            "H265")
                                clear
                                echo -e $stop
                                VENC="$ENC265 ! video/x-h265"
                                break
                                ;;
                            "VP8")
                                clear
                                echo -e $stop
                                VENC='vp8enc ! video/x-vp8'
                                break
                                ;;
                            "VP9")
                                clear
                                echo -e $stop
                                VENC='vp9enc ! video/x-vp9'
                                break
                                ;;
                            "Motion JPEG")
                                clear
                                echo -e $stop
                                VENC="$JPEGENC ! image/jpeg"
                                break
                                ;;
                            "Quit")
                                return=1
                                break
                                ;;
                            *) echo "invalid option $REPLY";;
                        esac
                    done
                    if [ $return == 0 ]
                    then
                        gst-launch-1.0 videotestsrc ! ${VENC}, mapping=$MAPPING ! rtspsink service=$PORT
                    fi
                done
                PS3='Select the media type to stream: '
                break
                ;;
            # Audio options
            "Audio")
                clear
                PS3='Select the encoding type for audio stream: '
                return=0
                while [ $return -lt 1 ]
                do
                    # Pipelines have no streaming limit. Allow user to stop manually
                    echo -e "\nSelect a Audio encoding format. Stream will start automatically." 
                    echo -e "Launch the rtsp_client.sh in a new terminal after the stream has started. \n"
                    select aud in "${audio_options[@]}"
                    do
                        case $aud in
                            "AAC")
                                clear
                                echo -e $stop
                                AENC='voaacenc ! audio/mpeg'
                                break
                                ;;
                            "AC3")
                                clear
                                echo -e $stop
                                AENC='avenc_ac3 ! audio/x-ac3'
                                break
                                ;;
                            "PCMU")
                                clear
                                echo -e $stop
                                AENC='mulawenc ! audio/x-mulaw'
                                break
                                ;;
                            "PCMA")
                                clear
                                echo -e $stop
                                AENC='alawenc ! audio/x-alaw'
                                break
                                ;;
                            "OPUS")
                                clear
                                echo -e $stop
                                AENC='opusenc ! audio/x-opus'
                                break
                                ;;
                            "Quit")
                                return=1
                                break
                                ;;
                            *) echo "invalid option $REPLY";;
                        esac
                    done
                    if [ $return == 0 ]
                    then
                        gst-launch-1.0 audiotestsrc ! ${AENC}, mapping=$MAPPING ! rtspsink service=$PORT
                    fi
                done
                PS3='Select the media type to stream: '
                break
                ;;
            # Allow change in port and stream mapping
            "Stream from file")
                clear
                PS3='Select an option: '
                return=0
                while [ $return -lt 1 ]
                do
                    # Pipelines have no streaming limit. Allow user to stop manually
                    echo -e "\nSelect an option. Stream will start automatically." 
                    echo -e "Launch the rtsp_client.sh in a new terminal after the stream has started. \n"
                    select fil in "${file_options[@]}"
                    do
                        case $fil in
                            "Start stream from file")
                                clear
                                if [ ! -f "$FILE" ]; then
                                    echo -e "\nDidn't find mp4 file. Creating mp4 file for streaming\n"
                                    gst-launch-1.0 -e videotestsrc is-live=true num-buffers=$VID_BUFFERS ! queue ! $ENC264 ! h264parse ! queue ! qtmux0. \
                                    audiotestsrc is-live=true num-buffers=$AUDIO_BUFFERS ! audioconvert ! audioresample ! voaacenc ! aacparse ! queue ! \
                                    qtmux ! filesink location=$FILE
                                fi
                                echo -e $stop
                                gst-launch-1.0 rtspsink name=sink service=$PORT appsink0::sync=true appsink1::sync=true qtdemux name=demux filesrc location=$FILE ! demux. \
                                            demux. ! queue ! aacparse ! audio/mpeg,mapping=$MAPPING ! sink. \
                                            demux. ! queue ! h264parse ! video/x-h264,mapping=$MAPPING ! sink.
                                break
                                ;;
                            "Set file path")
                                echo -e "\nInput file path to recording or press Enter to use default ($FILE): "
                                read FILE
                                if [ -z "$FILE" ]
                                then
                                    FILE=$RESOURCES_DIR/test_audio_video.mp4
                                fi
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
                PS3='Select the media type to stream: '
                break
                ;;
            "Change PORT/MAPPING")
                echo -e "\nInput port number or press Enter to use default (5000): "
                read PORT
                if [ -z "$PORT" ]
                then
                    PORT=5000
                fi
                echo -e "\nInput mapping name or press Enter to use default(/stream1): "
                read MAPPING
                if [ -z "$MAPPING" ]
                then
                    MAPPING=/stream1
                fi
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
