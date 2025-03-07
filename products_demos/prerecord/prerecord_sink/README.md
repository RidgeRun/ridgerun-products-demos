# GstPreRecord

The GstPrerecord element allows you to record video before and after and event happens, and trigger a video recording saving a time window before the trigger. If you want to know what happened before an event, the prerecord allows you to keep a time window in yuor final recording. This demo showcases how you can achieve this thanks to GstD.

Elements required:

* [GstPreRecord](https://developer.ridgerun.com/wiki/index.php/GStreamer_pre-record_element)

Project required

* [GstD](https://developer.ridgerun.com/wiki/index.php/GStreamer_Daemon)

## Contents

This folder contains the following examples

* multiple_recording.sh: Automatically generates multiple recordings each time a trigger is activated.     
* interactive_recording.sh: Allows user interaction; pressing the space bar triggers an extension of the recording duration.

## How to use it

First, ensure that gstd is running in the background as follows.

> gstd &

Execute the desire script.

### Multple recording example

Execute the script as follows

> ./multiple_recording.sh

### Interactive recording example

Start the script:

> ./interactive_recording.sh

* Press Space to activate trigger and extend the recording.
* Press C to change the pattern.
* Press Esc key to finish the demo.

### NVIDIA Jetson Platform

For executing the demo in a NVIDIA Jetson Platform use the following flag:

```
> ./multiple_recording.sh -j

or

> ./interactive_recording.sh -j
```

## Expected output.

Once you finish executing the demo, you will find a folder with the generated videos where you can play it with VLC or GStreamer.

> vlc output/recording_%Y-%m-%d_%H:%M:%S%z.mp4

# Troubleshooting

The first level of debug to troubleshoot a failing evaluation binary is to inspect GStreamer debug output.

> GST_DEBUG=2 ./interactive_recording.sh

If the output doesn't help you figure out the problem, please contact support@ridgerun.com with the output of the GStreamer debug and any additional information you consider useful.


