# GstSEI

GstSEI is a plugin with three elements, that allow inserting and extracting metadata into and from a H264 or H265 encoded stream.

You can find more information about GstSEI in our developer's wiki:

[GstSEI developer's wiki](https://developer.ridgerun.com/wiki/index.php?title=GstSEIMetadata)

You can purchase GstSEI product at:

[RidgeRun's store](https://shop.ridgerun.com/products/gstseimetadata?variant=40918489366715)

# Instructions

In this directory you will find a small collection of example uses for the GstSEI element. Most of these examples do not require any aditional hardware (such as camera sensors).
For more advanced use cases you can inquire at [RidgeRun support](http://www.ridgerun.com/contact).

Tests do not require a specific network as these are set to run on the machine's local network (```ip: 127.0.0.1```).

There are examples of how to use the element in a GStreamer application. To build these examples run the following command:
```
make
```
This will generate the executable files ```inject_out``` and```extract_out```. There is also the option to build each of the examples individually:
```
# Metadata injection example
make inject
# Metadata extraction example
make extract
```
To clean the built examples run:
```
make clean
```

## Jetson Platform

To build the examples for Jetson Platform use the following command:

```
make jetson
```

## Example list

The directory includes the following examples within it:
* Recording meta examples
    - Record a video file and insert SEI metadata (H264/H265)
    - Replay a video file and extract SEI metadata (H264/H265)
* GStreamer C examples
    - Send UDP stream with updating SEI metadata in each buffer.
    - Receive UDP stream, extract SEI metadata and overlay metadata on video.
* GstD python example
    - Connect and wait for the "new SEI metadata" signal

## Recording meta example

This example shows how SEI metadata can be added to video stream and saved to a file. The metadata can then be extracted and displayed.

To run this example, use the following command:

```
./recording_meta_example.sh
```

A menu will display the option to <code>Stream to file</code> which will create a small recording with the metadata. The user will be asked to select an encoding format (h264/h265) for the video stream.

Once the video has been created, the metadata can be extracted using the <code>View file and extract meta</code> option. Select the encoding type used to create the video. This will display the video stream and also print in the terminal the found metadata.

The default metadata string is "Hello world". The user can also modify this string to any other desired string with the <code>Change metadata string</code> option. In order to see the new string, create the recording again.

### Jetson Platform

For executing the demo in a Jetson Platform use the following flag:

```
./recording_meta_example.sh -j 
```

## GStreamer C example

The GStreamer application shows how to add updating metada strings to a video stream. This metadata can then be extracted on the viewed on the client application.

Start the sending pipeline, run:
```
./inject_out
```

Start the receiving pipeline in a second terminal:
```
./extract_out
```

The example starts a UDP video stream with the current time and date set to be the metadata. The receiving pipeline will display the video feed with a text overlay. This overlay is the extracted metadata received from the stream. For easier viewing, the metadata string is also printed to the terminal.

## GstD python example

This example shows how to use GstSEI in a python script with GstD. This allows to easily manage signals generated by the plugins.

**NOTE:** This example needs GStreamer Daemon to be installed on the current machine.

To run this example, start GstD on a terminal:
```
gstd -p 5000 -n 2
```

On a second terminal, run the python script:
```
python3 ./sei_python.py
```

The example will automatically create and start UDP Sender/Receiver pipelines. The example uses the <code>seimetatester</code> to add new metadata strings to the stream.

The <code>seiextract</code> plugin generates a signal when new metadata is received. This signal is connected to on a secondary GstD port to avoid blocking the rest of the application.

GstD allows for more information to be displayed for debugging. This information can be hard to follow and is not necessary in most cases. To view the debug information set the <code>logging</code> variable in the code to <code>DEBUG</code>.

### Jetson Platform

You can execute the python example for Jetson Platform as follows:

```
python3 sei_python.py -j 
```

or

```
python3 sei_python.py --jetson 
```
