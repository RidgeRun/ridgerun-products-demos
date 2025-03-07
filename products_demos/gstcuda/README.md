## GstCUDA Library examples

GstCUDA is a library that also offers a GStreamer plugin that enables the development of custom elements capable or running CUDA algorithms. 

In this folder you can find the examples for cudafilter and cudamuxer with the following structure.

algorithm.mk
README.md
    ├── cudafilter_algorithms
    │   ├── gray-scale-filter
    │   │   ├── Makefile
    │   │   ├── README.md
    │   │   ├── gray-scale-filter.cu
    │   │   ├── gray_scale_filter.sh
    │   │   └── test-pipelines.txt
    │   ├── median-filter
    │   │   ├── Makefile
    │   │   ├── README.md
    │   │   ├── median-filter.cu
    │   │   └── median_filter.sh
    │   ├── memcpy
    │   │   ├── Makefile
    │   │   ├── README.md
    │   │   ├── memcpy.cu
    │   │   └── memcpy_filter.sh
    │   └── pinhole
    │       ├── Makefile
    │       ├── README.md
    │       ├── pinhole.cu
    │       └── pinhole_filter.sh
    └── cudamux_algorithms
        └── mixer
            ├── Makefile
            ├── README.md
            ├── mixer.cu
            └── mixer_.sh

Inside of each folder you can find the instructions to build the binary and then you can execute the respective script. More details on how to execute the script with the --help flag.

./<NAME_OF_SCRIPT>.sh --help

For each example you need to have installed the following dependencies in a Jetson Platform:

* [GstCUDA developer's wiki](https://developer.ridgerun.com/wiki/index.php/GstCUDA)
* [GStreamer Daemon developer's wiki](https://developer.ridgerun.com/wiki/index.php/GStreamer_Daemon)

# Troubleshooting

The first level of debug to troubleshoot a failing evaluation binary is to inspect GStreamer debug output.

> GST_DEBUG=2 ./<NAME_OF_SCRIPT>.sh

If the output doesn't help you figure out the problem, please contact support@ridgerun.com with the output of the GStreamer debug and any additional information you consider useful.

