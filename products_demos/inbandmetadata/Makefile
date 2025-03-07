FLAGS=-Wall
PKG_CONFIG=`pkg-config --cflags --libs gstreamer-1.0`
CC= gcc
CODE_SOURCE = gstmetademo.c
EXECUTABLE = gstmetademo

default: all

all: $(EXECUTABLE)

$(EXECUTABLE): $(CODE_SOURCE)
	$(CC) $(FLAGS) $^ -o $@ $(PKG_CONFIG)

clean:
	rm $(EXECUTABLE)
