# Copyright (C) 2024 RidgeRun, LLC (http://www.ridgerun.com)
# All Rights Reserved.
#
# The contents of this software are proprietary and confidential to RidgeRun,
# LLC.  No part of this program may be photocopied, reproduced or translated
# into another programming language without prior written consent of
# RidgeRun, LLC.  The user is free to modify the source code after obtaining
# a software license from RidgeRun.  All source code changes must be provided
# back to RidgeRun without any encumbrance.

from pygstc.gstc import *
from pygstc.logger import *
import time

# Set logging to DEBUG to get more information on the pipeline state 
logging = 'ERROR'

# Amount of metadata buffers to insert
META_COUNT = 200

# Create pipeline descriptions
send_pipeline = 'metasrc name=meta ! meta/x-klv ! mpegtsmux name=mux ! ' \
        + 'udpsink host=127.0.0.1 port=5555 videotestsrc is-live=true ! ' \
        + 'x264enc ! h264parse ! queue ! mux.'

receive_pipeline = 'udpsrc address=127.0.0.1 port=5555 ! tsdemux name=demux ' \
        + 'demux. ! h264parse ! avdec_h264 ! queue ! videoconvert ! ' \
        + 'autovideosink sync=false async-handling=true demux. ! ' \
        + 'queue max-size-time=4000000000 max-size-buffers=200 ! ' \
        + 'capsfilter caps="meta/x-klv" ! metasink sync=true async=true'

# Create a custom logger that logs into stdout
gstd_logger = CustomLogger('inband_meta_example', loglevel=logging)
# Create the client and pass the logger as a parameter
gstd_client = GstdClient(logger=gstd_logger)

print("\nLog level is: {log_level}\n".format(log_level=logging))
print("\nCreate pipelines\n")
gstd_client.pipeline_create('sender', send_pipeline)
gstd_client.pipeline_create('receiver', receive_pipeline)

print("\nPlaying pipelines\n")
gstd_client.pipeline_play('sender')
time.sleep(2)
gstd_client.pipeline_play('receiver')

# Wait for receiver to playback
time.sleep(5)

count = 0
# Allow the stream to play
while (count < META_COUNT):
    gstd_client.element_set('sender', 'meta', 'metadata', f"Current count is: {count}")
    count = count + 1

# Wait time to allow receiver to display meta received
time.sleep(20)

#Stopping and deleting pipelines to enable rerun of example
print("\nStopping and deleting pipelines\n")
gstd_client.pipeline_stop('receiver')
gstd_client.pipeline_delete('receiver')

gstd_client.pipeline_stop('sender')
gstd_client.pipeline_delete('sender')

print("\n Closing ....")
