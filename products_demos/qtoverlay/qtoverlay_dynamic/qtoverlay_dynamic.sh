#!/bin/bash

END_MSJ="Finished stream"

trap "gstd-client pipeline_delete p1; echo -e $END_MSJ; exit" SIGHUP SIGINT SIGTERM

gstd-client pipeline_create p1 videotestsrc pattern=18 is-live=true name=src ! videoconvert ! video/x-raw,width=1920,height=1080  ! qtoverlay name=ql qml=logos.qml ! videoconvert ! autovideosink

update_time() {

	while true;
	do
		current_date=$(date +"%a-%b-%e-%Y %H:%M:%S")
		gstd-client element_set p1 ql qml-attribute "labelMain.text:$current_date"
	done
}



# Run pipeline
gstd-client pipeline_play p1
update_time &
pid=$!
active=true
while [ "$active" = true ];
do
	read -s -n1 key
	if [[ $key == $'\e' ]]; then
		kill $pid
		break
	fi
done

gstd-client pipeline_stop p1
gstd-client pipeline_delete p1
kill $pid
