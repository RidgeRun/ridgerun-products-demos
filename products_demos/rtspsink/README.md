# GstRTSPsink

The RTSP Sink is a GStreamer sink element which permits high performance streaming to multiple computers using the RTSP protocol. This element leverages previous logic from RidgeRun's RTSP server but providing the benefits of a GStreamer sink element like great flexibility to integrate into applications and easy gst-launch based testing. With GStreamer RTSP Sink multiple streams can be achieved simultaneously using any desired combination. This means that within a single GStreamer RTSP server pipeline you can stream multiple videos, multiple audios and multiple audio+video, each one to a different client and a different mapping. In the examples section, different streaming possibilities are shown using the GStreamer RTSP.

You can find more information about GstRTSPsink in our developer's wiki:

[GstRTSPsink developer's wiki](https://developer.ridgerun.com/wiki/index.php/GstRtspSink)

You can purchase GstRTSPsink product at:

[RidgeRun's store](https://shop.ridgerun.com/products/gstreamer-multi-stream-mulit-channel-rtsp-server-element)

# Instructions

In this directory you will find a small collection of example uses for the GstRTSPsink element. Most of these examples do not require any aditional hardware (such as camera sensors).
For more advanced use cases you can inquire at [RidgeRun support](http://www.ridgerun.com/contact).

Tests do not require a specific network as these are set to run on the machine's local network (```ip: 127.0.0.1```).

There are examples of how to use the element in a Gstreamer application. To build these examples run the following command:
```
make
```
This will generate the executable files ```parse_launch_out```,```manual_link_out``` and ```av_launch_out```. There is also the option to build each of the examples individually:
```
# Manual link example
make manual
# Parse launch
make parse 
# Audio + Video example 
make audio_video
```
To clean the built examples run:
```
make clean
```

**NOTE**: To launch these examples you will need to open 2 seperate terminals. One terminal is for the transmiting apps and the other will launch the rtsp clients.

## Jetson Platform

To build the examples for jetson platform use the following command.

```
make jetson
```

## Example list

The directory includes the following examples within it:
* Simple examples
    - Encode and transmit a video stream (MPEG4/H264/H265/VP8/VP9/Motion JPEG)
    - Encode and transmit an audio stream (AAC/AC3/PCMU/PCMA/OPUS)
    - Encode and transmit an audio+video stream from a mp4 file source
* Gstreamer C examples
    - Manually linking a video streaming pipeline
    - Parse a video streaming pipeline from a string
    - Launch an audio+video stream with a secondary video stream option. This stream needs authentication by receiving client

## Simple examples

These are a collection of video and audio pipelines that use different encodings and are sent through an RTSP stream.
To launch these examples you will need to open 2 seperate terminals. 

In terminal A run:
```
./simple_examples.sh
```
This should start a menu which allows you to select which type of media (audio or video) to stream. Once selected, the user will be presented with encoding types available for use.
Upon selecting the encoding type, the pipeline will launch and a stream will start automatically. To stop this stream use `Ctrl+C` to stop the pipe.

There is the option ```Stream from file```. This allows to stream an ***mp4*** file through RTSP. A default recording can be found in this repository. The user can set the example to stream from other files if they wish to stream another file.

**NOTE**: The receiving pipeline expects to receive a stream from ***mp4*** files with **audio and video**. If the streamed ***mp4*** does not have both, the client will fail. 

A client is needed to receive the stream. There are different RTSP clients available but to reduce complexity, a Gstreamer client pipeline is used. 

In terminal B run:
```
./rtsp_client.sh
```
To start the client select the ```Launch rtsp client``` option. This client **MUST** be launched after the RTSP stream has started or the client will fail. If a stream is found,
the client will connect and display the incoming video or play the audio. The client does NOT allow to save the stream to a file.

In each script there is an option to change the ```PORT``` and ```MAPPING``` for the stream. These options should match in the RTSP client and RTSP stream. If these do not match, the RTSP client will fail to connect.

If the ```Stream from file``` example is currently streaming, the ```Launch rtsp client for Stream from file``` option should be used. This client has a pipeline that is specific for this type of stream.

### Jetson Platform

For executing the demo in a Jetson Platform use the following flag:

```
./simple_examples.sh -j 
```

## Gstreamer C application examples

The Gstreamer applications show how to build a pipeline for the GstRTSPsink element in C. This pipeline will send a video output to a client. The examples are also a showcase of different methods to parse each of
the elements in the pipeline. These differences do not have a major impact on the stream quality.

In terminal A run:
```
./manual_link
```
or
```
./parse_launch
```
A client is needed to receive the stream. There are different RTSP clients available but to reduce complexity, a Gstreamer client pipeline is used.

In terminal B run:
```
./rtsp_client.sh
```
To start the client select the ```Launch rtsp client``` option. This client **MUST** be launched after the RTSP stream has started or the client will fail. If a stream is found,
the client will connect and display the incoming video or play the audio. The client does NOT allow to save the stream to a file.

In the client script there is an option to change the ```PORT``` and ```MAPPING``` for the stream. These options should match in the RTSP client and RTSP stream. If these do not match, the RTSP client will fail to connect.

## Audio + video example

This example shows how to stream audio and video at the same time in a single stream. The stream has authentication in the form of a ```USER/PASSWORD``` which needs to be provided on the client side in order to view
the stream. 

To run this example, in terminal A run:
```
./av_launch_out
```
This will start the stream audio+video stream. This application also has the option to start a second stream that is sent to the same ```PORT``` but has its ```MAPPING``` is ```stream2```. This secondary stream is **video only**.
To enable the second stream, modify the previous command as follows:
```
./av_launch_out --camera
```  
**NOTE**: this second stream expects a USB webcam. The stream will fail if there are no cameras available.

To view the stream run the client script in terminal B:
```
./rtsp_client.sh
```
To start the client select the ```Launch rtsp client``` option. This client **MUST** be launched after the RTSP stream has started or the client will fail. If a stream is found,
the client will connect and display the incoming video or play the audio. The client does NOT allow to save the stream to a file.

In the client script there is an option to change the ```PORT``` and ```MAPPING``` for the stream. There is also the option to change the ```USER``` and ```PASSWORD```.
These options should match in the RTSP client and RTSP stream. If these do not match, the RTSP client will fail to connect
