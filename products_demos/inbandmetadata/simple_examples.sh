#!/bin/bash

PS3='Select the media type to stream: '
stop="\nTo stop stream press Ctrl + C \n"
options=("Video" "Audio" "Metadata Only" "Quit")
video_options=("H264 encoding" "H264 decoding" "H265 encoding" "H265 decoding" "Quit")
audio_options=("AAC encoding" "AAC decoding" "Quit")
metadata_options=("Simple Metadata Access" "Basic periodically metadata" "Verify metadata content" "Simultaneous metadata sending and receiving" "Change metadata insertion period" "Quit")
exit=0

export METADATA='hello_world'
export TIME_METADATA=The_current_time_is:%T
export FILE=metadata_video.ts
export TEXT=meta.txt
export SEC=1

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
                    select enc in "${video_options[@]}"
                    do
                        case $enc in
                            "H264 encoding")
                                clear
                                echo -e $stop
                                META_TYPE=$TIME_METADATA
                                ENCODE='264'
                                RUN_ENCODE=true
                                break
                                ;;
                            "H264 decoding")
                                clear
                                echo -e $stop
                                DECODE='h264'
                                RUN_DECODE=true
                                break
                                ;;
                            "H265 encoding")
                                clear
                                echo -e $stop
                                META_TYPE=$METADATA
                                ENCODE='265'
                                RUN_ENCODE=true
                                break
                                ;;
                            "H265 decoding")
                                clear
                                echo -e $stop
                                DECODE='h265'
                                RUN_DECODE=true
                                break
                                ;;
                            "Quit")
                                return=1
                                break
                                ;;
                            *) echo "invalid option $REPLY";;
                        esac
                    done
                    if [[ $RUN_DECODE == true ]]; then
                        RUN_DECODE=false
                        gst-launch-1.0 -e filesrc location=$FILE ! tsdemux name=demux demux. ! queue ! ${DECODE}parse ! \
                        'video/x-'${DECODE}', stream-format=byte-stream, alignment=au' ! avdec_${DECODE} ! \
                        autovideosink sync=false async-handling=true demux. ! queue ! 'meta/x-klv' ! \
                        metasink sync=true async=true
                    fi
                    if [[ $RUN_ENCODE == true ]]; then
                        RUN_ENCODE=false
                        gst-launch-1.0 -e metasrc metadata=${META_TYPE} period=1 ! 'meta/x-klv' ! mpegtsmux name=mux ! \
                        filesink sync=false async=true location=$FILE videotestsrc is-live=true ! \
                        'video/x-raw,format=(string)I420,width=320,height=240,framerate=(fraction)30/1' ! \
                        x${ENCODE}enc ! mux.
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
                    select aud in "${audio_options[@]}"
                    do
                        case $aud in
                            "AAC encoding")
                                clear
                                echo -e $stop
                                gst-launch-1.0 -e metasrc metadata=$METADATA ! 'meta/x-klv' ! mpegtsmux name=mux ! filesink sync=false async=true location=$FILE audiotestsrc num-buffers=300 \
                                is-live=true ! 'audio/x-raw, channels=(int)1, rate=(int)48000, layout=(string)interleaved, format=(string)S16LE' ! faac ! 'audio/mpeg, mpegversion=(int)4, \
                                channels=(int)1, rate=(int)48000, stream-format=(string)adts, profile=(string)lc' ! mux.

                                break
                                ;;
                            "AAC decoding")
                                clear
                                echo -e $stop
                                gst-launch-1.0 -e filesrc location=$FILE ! tsdemux name=demux demux. ! faad ! alsasink demux. ! queue ! 'meta/x-klv' ! metasink
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
            # Only Metadata options
            "Metadata Only")
                clear
                PS3='Select metadata only test: '
                return=0
                while [ $return -lt 1 ]
                do
                    # Pipelines have no streaming limit. Allow user to stop manually
                    echo -e "\nSelect a test:" 
                    select mtd in "${metadata_options[@]}"
                    do
                        case $mtd in
                            "Simple Metadata Access")
                                clear
                                echo -e $stop
                                gst-launch-1.0 -e metasrc metadata=$TIME_METADATA period=1 ! 'meta/x-klv' ! filesink sync=false async=true location=$TEXT
                                break
                                ;;
                            "Basic periodically metadata")
                                clear
                                echo -e $stop
                                gst-launch-1.0 -e metasrc metadata="$TIME_METADATA" period=$SEC ! 'meta/x-klv' ! filesink sync=false async=true location=$TEXT
                                break
                                ;;
                            "Verify metadata content")
                                clear
                                echo -e $stop
                                gst-launch-1.0 -e filesrc location=$TEXT ! 'meta/x-klv' ! metasink
                                break
                                ;;
                            "Simultaneous metadata sending and receiving")
                                clear
                                echo -e $stop
                                gst-launch-1.0 -e metasrc metadata="$TIME_METADATA" period=$SEC ! 'meta/x-klv' ! metasink
                                break
                                ;;
                            "Change metadata insertion period")
                                clear
                                echo -e "\nInput new metadata insertion period or press Enter to use default (2)s: "
                                read SEC
                                if [ -z "$SEC" ]
                                then
                                    SEC=2
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
            "Quit")
                exit=1
                break
                ;;
            *) echo "invalid option $REPLY";;
        esac
    done
done
