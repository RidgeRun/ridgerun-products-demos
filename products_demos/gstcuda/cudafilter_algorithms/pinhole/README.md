# Example GstCUDA external algorithm

This area provides an example on how to easily generate an external
algorithm to be consumed with cudafilter. The steps are as follows:

## Process
* Build gst-cuda
* Add a new makefile to your project that includes the helper makefile. E.g.:
```
# Including GstCuda helper makefile
include /home/mgruner/RidgeRun/gst-cuda/tools/algorithm.mk
```
* Add the sources to your project
* Build
```
make
```

## Targets

The helper makefile provides to targets
* make
* make clean

## Customization

Even though the helper makefile should typically be enough for most
project requirements, there are some variables that may be adjusted
to control the flow of the build.

Define this variables after the makefile inclusion.

* GST_CUDA_LIB: Use this variable if an alternative GstCuda library
  name is used. The default is libgstcuda-1.0.so

* GST_CUDA_LIBDIR: Use this variable if the GstCuda library is to be
read from a location other than the installed by the project. The
default is typically /usr/lib/aarch64-linux-gnu

* GST_CUDA_INCDIR: Use this variable if you are reading headers from a
location other than the used by the project. The default is
typically /usr/include/gstreamer-1.0/sys/cuda/

* CUDA_NVCC: Use this variable to select an alternative NVCC
than the one found at configure time. The default is tyically
/usr/local/cuda-8.0/bin/nvcc

* CUDA_FLAGS: Use this variable if you would like to completely
override the nvcc configuration found at build time. The default is
typically "-m64 -O3 -arch=sm_53 -Xcompiler -Wall -Xcompiler -Wextra
--ptxas-options=-v # -Xcompiler -fPIC -DPIC -D_FORCE_INLINES"

* CUDA_EXTRA_CFLAGS: Use this variable if you would like to append
nvcc compiler flags to the ones provided by CUDA_FLAGS.

* GST_INCDIR: Use this variable if you would like to use a different
set of GStreamer headers from the ones you used during the GstCuda
build. The default is -I/gst-libs -I/usr/include/gstreamer-1.0
-I/usr/lib/aarch64-linux-gnu/gstreamer-1.0/include
-I/usr/include/glib-2.0 -I/usr/lib/aarch64-linux-gnu/glib-2.0/include

* CXXFLAGS Use this variable if you would like to completely override
the compiler configuration. The default is typically -std=c++11
-I/usr/include/gstreamer-1.0 -I/gst-libs -I/usr/include/gstreamer-1.0
-I/usr/lib/aarch64-linux-gnu/gstreamer-1.0/include -I/usr/include/glib-2.0
-I/usr/lib/aarch64-linux-gnu/glib-2.0/include -I/usr/include/glib-2.0
-I/usr/lib/aarch64-linux-gnu/glib-2.0/include -D_GLIB_TEST_OVERFLOW_FALLBACK

* EXTRA_CXXFLAGS: Use this variable if you would like to append compiler
flags to the ones provided by CXXFLAGS.

* LDFLAGS: Use this variable if you would like to completely override
the linker configuration. The default is typically --shared
-L/usr/lib/aarch64-linux-gnu -lgstcuda-1.0 -Wno-deprecated-gpu-targets

* EXTRA_LDFLAGS: Use this variable if you would like to append linker
flags to the ones provided by LDFLAGS

* ALGORITHM: Use this variable if you would like to use a custom name
for the algorithm. By default the makefile will use the directory name
as the name for the shared object

* SUFFIX: Use this variable if you are using a file suffix other than
"cu".  The makefile will compile all the *.$(SUFFIX) in the
directory. The default is cu

* SOURCES: Uncomment this variable if you would like to pass in a
custom list of source files. By default the makefile will look for
all the source files in the current directory.
