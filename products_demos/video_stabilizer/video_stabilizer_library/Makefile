CC=g++
CFLAGS=`pkg-config --cflags --libs gstreamer-1.0 opencv4 rvs`
RAW_GYRO_INPUT_FILE = "../samples/raw_gyro_data.csv"
TEST_VIDEO = "../samples/non_stablized.mp4"
EXAMPLES = -DRAW_GYRO_INPUT_FILE="\"$(RAW_GYRO_INPUT_FILE)\"" -DTEST_VIDEO="\"$(TEST_VIDEO)\""

default: all

all: rvs-complete-concept

rvs-complete-concept: complete-code-example.cpp
	$(CC) $^ $(CFLAGS) $(EXAMPLES) $(CUDA_FLAGS) $(OPENCL_FLAGS) -o $@

clean:
		rm rvs-complete-concept

