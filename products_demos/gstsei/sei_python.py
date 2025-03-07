# Copyright (C) 2024 RidgeRun, LLC (http://www.ridgerun.com)
# All Rights Reserved.
#
# The contents of this software are proprietary and confidential to RidgeRun,
# LLC.  No part of this program may be photocopied, reproduced or translated
# into another programming language without prior written consent of
# RidgeRun, LLC.  The user is free to modify the source code after obtaining
# a software license from RidgeRun.  All source code changes must be provided
# back to RidgeRun without any encumbrance.

import argparse
from pygstc.gstc import *
from pygstc.logger import *
import threading
import time

def parse_args():
    parser = argparse.ArgumentParser(
        description=" ",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--jetson','-j',
                        help='Enable NVIDIA Jetson mode',
                        action='store_true',default=False)
    
    return parser.parse_args()

# Set logging to DEBUG to get more information on the pipeline state 
logging = 'ERROR'

# Amount of signal checks done
SIGNAL_CHECKS = 10
args=parse_args()
if args.jetson:
    enc264='nvvidconv ! nvv4l2h264enc'
else:
    enc264='x264enc'

def signal_connect_handler():
    signal_logger = CustomLogger('gstsei_signal', loglevel=logging)
    # Create the client in a different port (5001) to avoid congestion
    signal_client = GstdClient(logger=signal_logger, port=5001)
    # Loop allows to capture metadata signal multiple times
    checking = 0

    while (checking < SIGNAL_CHECKS):
        ret_val = signal_client.signal_connect('receiver', 'extract', 'new-metadata')
        checking = checking + 1
        print(f"\nMetada signal received. Signal contains:\n{ret_val}")
# Create pipeline descriptions
send_pipeline = 'videotestsrc is-live=true ! ' + enc264 + ' ! seimetatimestamp ! ' \
        + 'seiinject ! rtph264pay ! capsfilter name=cf ! ' \
        + 'udpsink host=127.0.0.1 port=5050'

print(send_pipeline)

receive_pipeline = 'udpsrc port=5050 ! capsfilter name=cf ' \
        + 'caps="application/x-rtp,media=video,clock-rate=90000,encoding-name=H264" ! ' \
        + 'rtph264depay ! h264parse ! seiextract signal-new-metadata=true name=extract ! ' \
        + 'fakesink'

# Create a custom logger that logs into stdout
gstd_logger = CustomLogger('gstsei_example', loglevel=logging)
# Create the client and pass the logger as a parameter
gstd_client = GstdClient(logger=gstd_logger)

print("\nLog level is: {log_level}\n".format(log_level=logging))
print("\nCreate pipelines\n")
gstd_client.pipeline_create('sender', send_pipeline)
gstd_client.pipeline_create('receiver', receive_pipeline)

print("\nPlaying pipelines\n")
gstd_client.pipeline_play('receiver')

time.sleep(2)
gstd_client.pipeline_play('sender')

# Using the thread allows to wait for signal without blocking 
# the rest of the application
signal_connect_thread = threading.Thread(target=signal_connect_handler)
signal_connect_thread.start()

# Waiting for the new meta signals. In a production enviroment
# this time could be used to continue with another process.
time.sleep(10)

#Stopping and deleting pipelines to enable rerun of example
print("\nStopping and deleting pipelines\n")
gstd_client.pipeline_stop('receiver')
gstd_client.pipeline_delete('receiver')

gstd_client.pipeline_stop('sender')
gstd_client.pipeline_delete('sender')

print("\n Closing ....")
