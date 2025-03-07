# GstQtOverlay + GstPreRecord

The GstPreRecord element allows you to record video before and after and event happens, and trigger a video recording saving a time window before the trigger. If you want to know what happened before an event, the prerecord allows you to keep a time window in your final recording. Meanwhile, the GstQtOverlay element allows you to overlay QML components on the video stream.

This demo demonstrates how to use GstQtOverlay to display a "Rec" signal while recording whenever the user activates the trigger, utilizing GstD.

Elements required:

* [GstPreRecord](https://developer.ridgerun.com/wiki/index.php/GStreamer_pre-record_element)
* [GstQtOverlay](https://developer.ridgerun.com/wiki/index.php/GStreamer_Qt_Overlay_for_Embedded_Systems?_gl=1*wmtav*_gcl_au*NDEyNTI5ODIzLjE3Mzc1ODQ2Nzk.)

Project required

* [GstD](https://developer.ridgerun.com/wiki/index.php/GStreamer_Daemon)

## Contents

This folder contains the following files:

* qtoverlay_interactive_recording.sh: The main script for the demo.
* qt_prerecord.qml: The QML file that displays the "Rec" GIF and text when the trigger is pressed and the logo of RidgeRun.
* rec.gif: The "Rec" animation used in the overlay.
* RWLogo.png: RidgeRun's logo used in the demo.

## How to use it

First, ensure that gstd is running in the background as follows.

> gstd &

Execute the script.

> ./qtoverlay_interactive_recording.sh

* Press Space to activate trigger and extend the recording.
* Press Esc key to finish the demo.

## Expected output.

Once you finish executing the demo, a folder will be created containing the recorded videos. You can play the recordings using VLC or GStreamer:

> vlc output/recording_%Y-%m-%d_%H:%M:%S%z.mp4

# Troubleshooting

The first level of debug to troubleshoot a failing evaluation binary is to inspect GStreamer debug output.

> GST_DEBUG=2 ./interactive_recording.sh

If the output doesn't help you figure out the problem, please contact support@ridgerun.com with the output of the GStreamer debug and any additional information you consider useful.


