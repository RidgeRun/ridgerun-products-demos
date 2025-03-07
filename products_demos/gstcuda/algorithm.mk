# 
# Copyright (C) 2017 RidgeRun, LLC (http://www.ridgerun.com)
# All Rights Reserved.
#
# The contents of this software are proprietary and confidential to RidgeRun,
# LLC.  No part of this program may be photocopied, reproduced or translated
# into another programming language without prior written consent of 
# RidgeRun, LLC.  The user is free to modify the source code after obtaining
# a software license from RidgeRun.  All source code changes must be provided
# back to RidgeRun without any encumbrance.
#

# GstCuda configuration time variables, do not override
prefix=/usr
exec_prefix=${prefix}

# User may override this variables
GST_CUDA_LIB ?= libgstcuda-1.0.so
GST_CUDA_LIBDIR ?= /usr/lib/aarch64-linux-gnu
GST_CUDA_INCDIR ?= ${prefix}/include/gstreamer-1.0

CUDA_NVCC ?= /usr/local/cuda/bin/nvcc
CUDA_FLAGS ?=  -m64 -O3 -Xcompiler -Wall -Xcompiler -Wextra --ptxas-options=-v -Xcompiler "-fPIC -DPIC" -D_FORCE_INLINES
CUDA_EXTRA_CFLAGS ?=

GST_INCDIR ?= $(filter -I% -D%, -I$(top_srcdir)/gst-libs -pthread -I/usr/include/gstreamer-1.0 -I/usr/include/aarch64-linux-gnu -I/usr/include/glib-2.0 -I/usr/lib/aarch64-linux-gnu/glib-2.0/include -DGST_USE_UNSTABLE_API -Wno-error=missing-include-dirs  -DG_THREADS_MANDATORY -DG_DISABLE_CAST_CHECKS $(GST_OPTION_CFLAGS))

CXXFLAGS ?= -std=c++11 -I$(GST_CUDA_INCDIR) $(GST_INCDIR) -D_GLIB_TEST_OVERFLOW_FALLBACK

EXTRA_CXXFLAGS ?=
LDFLAGS ?= --shared -L$(GST_CUDA_LIBDIR) -l$(patsubst lib%.so,%,$(GST_CUDA_LIB)) -Wno-deprecated-gpu-targets
EXTRA_LDFLAGS ?=

ALGORITHM ?= $(shell basename `pwd`).so
SUFFIX ?= cu
SOURCES ?= $(wildcard *.$(SUFFIX))
OBJECTS ?= $(patsubst %.$(SUFFIX),%.o,$(SOURCES))

# Verbose output
ifeq ("$(V)","1")
Q :=
vecho = @echo
else
Q := @
vecho = @true
endif

.DEFAULT_GOAL := $(ALGORITHM)

.PHONY: debug clean

$(ALGORITHM):: debug $(OBJECTS)
	$(vecho) Linking $@ from $(OBJECTS)
	$(Q)$(CUDA_NVCC) -o $@ $(OBJECTS) $(LDFLAGS) $(EXTRA_LDFLAGS)

%.o: %.$(SUFFIX)
	$(vecho) Compiling $<
	$(Q)$(CUDA_NVCC) -o $@ -c $< $(CUDA_FLAGS) $(CUDA_EXTRAFLAGS) $(CXXFLAGS) $(EXTRA_CXXFLAGS)

debug:
	$(vecho) =====================================
	$(vecho) GstCuda helper makefile configuration
	$(vecho) -------------------------------------
	$(vecho) GST_CUDA_LIB = $(GST_CUDA_LIB)
	$(vecho) GST_CUDA_LIBDIR = $(GST_CUDA_LIBDIR)
	$(vecho) GST_CUDA_INCDIR = $(GST_CUDA_INCDIR)
	$(vecho)
	$(vecho) CUDA_NVCC = $(CUDA_NVCC)
	$(vecho) CUDA_FLAGS = $(CUDA_FLAGS)
	$(vecho) CUDA_EXTRA_CFLAGS = $(CUDA_EXTRA_CFLAGS)
	$(vecho)
	$(vecho) GST_INCDIR = $(GST_INCDIR)
	$(vecho)
	$(vecho) CXXFLAGS = $(CXXFLAGS)
	$(vecho) EXTRA_CXXFLAGS = $(EXTRA_CXXFLAGS)
	$(vecho) LDFLAGS = $(LDFLAGS)
	$(vecho) EXTRA_LDFLAGS = $(EXTRA_LDFLAGS)
	$(vecho)
	$(vecho) ALGORITHM = $(ALGORITHM)
	$(vecho) SUFFIX = $(SUFFIX)
	$(vecho) SOURCES = $(SOURCES)
	$(vecho) OBJECTS = $(OBJECTS)
	$(vecho) ====================================
	$(vecho)

clean:
	$(vecho) Cleaning project
	$(Q)rm -f $(ALGORITHM)
	$(Q)rm -f $(OBJECTS)
	$(Q)rm -f *~
