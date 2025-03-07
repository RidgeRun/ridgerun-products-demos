#!/bin/bash

IP_ADDRESS=127.0.0.1
PORT=5000
MAPPING=/stream1
USER=anonymous
PASSWORD=secret

options=("Launch rtsp client" "Change PORT/MAPPING" "Change authentication USER/PASSWORD" "Quit")
PS3='Select an option: '
stop="\nTo stop stream press Ctrl + C \n"
return=0

clear
while [ $return -lt 1 ]
do
    # Pipelines have no streaming limit. Allow user to stop manually
    echo -e "\nLaunch the rtsp_client.sh after RTSP stream has started."
    echo -e "If the stream has not started or is stopped, rtsp client will stop with error."
    select opt in "${options[@]}"
    do
        case $opt in
            "Launch rtsp client")
                clear
                echo -e $stop
                gst-launch-1.0 playbin uri=rtsp://${USER}:${PASSWORD}@${IP_ADDRESS}:${PORT}/${MAPPING}
                break
                ;;
            "Change PORT/MAPPING")
                echo -e "\nInput port number or press Enter to use default (5000): "
                read PORT
                if [ -z "$PORT" ]
                then
                    PORT=5000
                fi
                echo -e "\nInput mapping name or press Enter to use default (/stream1): "
                read MAPPING
                if [ -z "$MAPPING" ]
                then
                    MAPPING=/stream1
                fi
                break
                ;;
            "Change authentication USER/PASSWORD")
                echo -e "\nInput user or press Enter to use default (anonymous): "
                read PORT
                if [ -z "$USER" ]
                then
                    USER=anonymous
                fi
                echo -e "\nInput password or press Enter to use default (secret): "
                read MAPPING
                if [ -z "$PASSWORD" ]
                then
                    PASSWORD=secret
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
