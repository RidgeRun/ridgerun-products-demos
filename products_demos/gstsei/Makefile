CC=gcc
CFLAGS=`pkg-config --cflags --libs gstreamer-1.0`

default: all

all: inject_out extract_out

inject_out: seiinject_video.c
		$(CC) $^ $(CFLAGS) -o $@

extract_out: seiextract_video.c
		$(CC) $^ $(CFLAGS) -o $@

clean:
		rm *_out
