# In-Band Metadata

GStreamer In-Band Metadata for MPEG Transport Stream is a set of plugins for the communication channel used to carry additional data or information related to a specific source, allowing receivers to extract and access this metadata, such as GPS location in video frames, while ensuring it does not affect the operation of receivers unable to interpret in-band metadata.

You can find more information about In-Band Metadata in our developer's wiki:

[In-Band Metadata developer's wiki](https://developer.ridgerun.com/wiki/index.php/GStreamer_In-Band_Metadata_for_MPEG_Transport_Stream)

You can purchase In-Band Metadata product at:

[RidgeRun's store](https://shop.ridgerun.com/products/gstreamer-in-band-metadata-support)

# Instructions

In this directory you will find a small collection of example uses for the In-Band Metadata element. Most of these examples do not require any aditional hardware (such as camera sensors).
For more advanced use cases you can inquire at [RidgeRun support](https://www.ridgerun.com/contact).

Tests do not require a specific network as these are set to run on the machine's local network (```ip: 127.0.0.1```).

There are examples of how to use the element in a GStreamer application. To build these examples run the following command:
```
make
```
This will generate the executable file ```gstmetademo```.

To clean the built examples run:
```
make clean
```
## Example list

The directory includes the following examples within it:
* Simple examples
    - Record a video file and insert metadata (H264/H265)
    - Replay a video file and extract metadata (H264/H265)
    - Record an audio file and insert metadata
    - Replay an audio file and extract metadata
    - Insert metadata into a text file
    - Insert metadata into a text file (user can update metadata insertion time period)
    - Extract metadata from text file
* GStreamer C example
    - Record audio+video file with updating metadata and text overlay.
* GstD python example
    - Send and receive an audio+video stream with updating metadata through UDP

# Simple examples

It is a collection of metadata mixed with video, audio or just metadata with different processes, such as encoding that saves the metadata in a file for later decoding.

To run these tests use these commands:
```
./simple_examples.sh
```
This should launch a menu that allows you to select which pipeline you want to test (Video, Audio, Metadata Only). The video and audio will insert metadata into the buffers and if you choose these options. These options bring up another menu showing the types of encoder and decoder available.

**NOTE:** The file recordings do not have a preset length. To stop end the recording process, press <code>CTRL+C</code>.

If the <code>Metadata Only</code> option is selected, options to test the metadata transport are shown. The metadata is saved to a <code>.txt</code> file. This metadata can later be extracted and displayed.

# GStreamer C application examples

The GStreamer apps show how to create a pipeline for the meta element in C. This pipeline inserts metadata and into a video and audio stream. The recorded video has a text overlay showing the inserted metadata.

In the terminal run:

```
./gstmetademo
```
The application also allows the user to change the duration of the recording and the output file name. For example:
```
./gstmetademo -o ./output.ts -l 5 # Recording is named output.ts file with 5s duration
```
To view a list of the options in detail, run:
```
./gstmetademo --help
```

To view only the video run:
```
gst-play-1.0 meta.ts
```

To play the video and extract the metadata, use the <code>simple_examples.sh</code> script. Select the <code>Video</code> -> <code>H264 decoding</code> option. This will display the recording and the metadata.

# GstD pyton example

This example shows how to use GStreamer In-band metadata in a python script with GstD. This allows to easily manage the pipelines and properties of each plugins.

**NOTE:** This example needs GStreamer Daemon to be installed on the current machine.

To run this example, start GstD on a terminal:
```
gstd
```

On a second terminal, run the python script:
```
python3 ./inbandmeta_python.py
```

The example will automatically create and start UDP Sender/Receiver pipelines. The example uses the <code>metasrc</code> to add new metadata strings to a video stream. The plugin <code>meta</code> property is set manually using the GstD <code>element_set</code> functionality.

The <code>metasink</code> plugin is used to extract the metadata buffers in the receiver pipeline. The receiver pipeline will display the video stream and print out the metadata buffers.

**NOTE:** The <code>metasink</code> plugin will display the metadata information in the first terminal, which is running the GStreamer Daemon.

GstD allows for more information to be displayed for debugging. This information can obscure relevant information and is not necessary in most cases. To view the debug information set the <code>logging</code> variable in the code to <code>DEBUG</code>.
