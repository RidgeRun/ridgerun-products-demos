# Spherical 360 Dynamic PTZ

This demo showcases how to use the stitching functionality with a projector and a spherical PTZ camera in a script using GSTD. The demo provides one mode of operation: using pre-recorded video files as input.

This reference design uses the following RidgeRun products:
* [GstCUDA developer's wiki](https://developer.ridgerun.com/wiki/index.php/GstCUDA)
* [GstStitcher developer's wiki](https://developer.ridgerun.com/wiki/index.php/Image_Stitching_for_NVIDIA_Jetson)
* [GstProjector developer's wiki](https://developer.ridgerun.com/wiki/index.php/RidgeRun_Image_Projector)
* [GstQtOverlay developer's wiki](https://developer.ridgerun.com/wiki/index.php/GStreamer_Qt_Overlay_for_Embedded_Systems)
* [GstRtspSink developer's wiki](https://developer.ridgerun.com/wiki/index.php/GstRtspSink)
* [Spherical Video PTZ developer's wiki](https://developer.ridgerun.com/wiki/index.php/Spherical_Video_PTZ)

# How to use

## Install Peoplenet model

Execute the script for the installation

> ./install_dep.sh

## Running the Script with Video Files

As default the demo will open a display using the default videos provided in the download.sh script.

> ./stitcher_ptz.sh

If you prefer to use your own pre-recorded video files as input, run the script as follows:

> ./stitcher_ptz.sh -m "<INPUT_1>.mp4 <INPUT_2>.mp4 <INPUT_3>.mp4"

Replace <INPUT_1>.mp4, <INPUT_2>.mp4, and <INPUT_3>.mp4 with the paths to your video files. The same PTZ controls (arrows for pan/tilt and more ) are available in this mode.

Also, is possible to start in the automatic mode with the following flag.

> ./stitcher_ptz.sh -a

To enable rtsp to send the stream to other computer, use the flag of -r with the selected port as follows.

> ./stitcher_ptz.sh -r 7000

When you don't want to use the display or avoid any error when not connected to one, then use the -n flag.

> ./stitcher_ptz.sh -n

This will open a window where you can interact with the PTZ controls:

- Pan and Tilt: Use the arrow keys to pan (left/right) and tilt (up/down).
- Zoom: Use the + and - keys to zoom in and out.
- Automatic mode: Use the key '0' to activate and desactivate the mode.
- Reset position: Use the key '1' to reset the position.
- Set new reset position: Use the key '2' in your new location to set the new position.
- Exit: You can finalize the demo with the "ESCAPE" key or with "Ctrl+C"

## NVIDIA Jetson Platform

For executing the demo in a NVIDIA Jetson Platform use the following flag:

```
./stitcher_ptz.sh -j 
```

## Receiving the rtsp stream 

For the rtsp feature, you can receive the stream with the following pipeline.

> IP_ADDRESS= <ip address of the board>
> PORT= <port selected with the rtsp flag -r>
> gst-launch-1.0 rtspsrc location=rtsp://$IP_ADDRESS:$PORT/stream1 ! decodebin ! queue ! videoconvert !  autovideosink sync=false 


