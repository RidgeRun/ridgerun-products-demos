# LibMISB Example Application

This application comes from the other usage applications, but applied in an scenario where you want to codify a JSON MISB file, then the coded decimal data from the output of the misb converter app will be used as the input for a GStreamer application, and then, stream the information over UDP by setting an IP Address and a port, to finally get the information in another GStreamer pipeline in a different terminal, using the network port set in the the execution of the GStreamer application.

### Links to buy RidgeRun products needed to run the example

LibMISB: https://shop.ridgerun.com/products/libmisb
GStreamer in-band metada support: https://shop.ridgerun.com/products/gstreamer-in-band-metadata-support

### Libraries

Please install the following libraries:
<pre>
sudo apt install -y gstreamer1.0-plugins-ugly gstreamer1.0-plugins-bad gstreamer1.0-libav libglib2.0-dev
</pre>

### How to execute the application

1. Run the following command to compile the libmisb_app.cpp file:
<pre>
g++ libmisb_app.c -o libmisb_app `pkg-config gstreamer-1.0 --cflags --libs glib-2.0 misb-0.0`
</pre>


2. Open a terminal to run the GStreamer pipeline that will get the stream remotely from the application. </br>
Run this command:
<pre>
gst-launch-1.0 -e udpsrc port=5000 ! 'video/mpegts, systemstream=(boolean)true, packetsize=(int)188' ! tsdemux name=demux demux. ! queue !  h264parse ! 'video/x-h264, stream-format=byte-stream, alignment=au' ! avdec_h264 ! autovideosink sync=false demux. ! queue ! 'meta/x-klv' ! metasink async=false
</pre>

3. Go back to the terminal where you compiled the libmisb_app.cpp file to run the GStreamer application. </br>
To run it you must provide an IP address, port and the JSON file path where that you want to encode and send through the streaming, for default we recommend to use 127.0.0.1 for the IP address, and 5000 for the port. The command to run is:
<pre>
./libmisb_app.cpp 127.0.0.1 5000 your/json/path/location/misb_ST0601_sample.json
</pre>

### Extra code

You will use this extra code to create a binary file from the incoming data that you can get from the ouptut fo the GStreamer pipeline. You can compile the code by using:
<pre>
gcc binary_file_creator.c -o binary_file_creator
</pre>

The usage is the following:
<pre>
binary_file_creator -n <name_of_binary_output.bin> <group of decimal values representation>
</pre>

Example:
<pre>
./binary_file_creator -n output.bin 6 14 43 52 2 11 1 1 14 1 3 1 1 0 0 0 44 2 8 0 4 89 249 174 32 34 168 3 9 77 73 83 83 73 79 78 48 49 4 6 65 70 45 49 48 49 5 2 113 194 15 2 194 33 65 1 17 1 2 164 125
</pre>

This will create a file called output.bin, to create the JSON file you can use:
<pre>
./misb-converter --decode -i output.bin -o output.json
</pre>
