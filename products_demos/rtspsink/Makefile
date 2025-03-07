CC=gcc
CFLAGS=`pkg-config --cflags --libs gstreamer-1.0`
jetson: CFLAGS += -DJETSON
default: all

all: parse_launch_out manual_link_out audio_video_out
jetson: all
parse_launch_out: rtspsink_parse_launch_example.c
		$(CC) $^ $(CFLAGS) -o $@

manual_link_out: rtspsink_manual_link_example.c
		$(CC) $^ $(CFLAGS) -o $@

audio_video_out: rtspsink_audio_video_example.c
		$(CC) $^ $(CFLAGS) -o $@

clean:
		rm *_out
