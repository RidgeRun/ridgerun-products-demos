# GstPreRecord

The GstPreRecord element is a filter that prerecords data continuously into a FIFO. You can set the FIFO size in milliseconds based on the amount of pre-recorded data you want to keep. When pre-recording, the pre-record element doesn't pass any buffer downstream. After the FIFO is filled, the oldest data in the FIFO is released as new data is added. When you want to start recording, you can trigger the pre-record element and it will pass the data in the FIFO downstream while adding new data to the end of the FIFO buffer so no data is lost. Eventually, the FIFO will be completely drained and the element will act as a pass-through. When the pipeline is stopped, the pre-record process can be repeated. 

You can find more information about GstPreRecord in our developer's wiki:

[GstPreRecord developer's wiki](https://developer.ridgerun.com/wiki/index.php?title=GStreamer_pre-record_element#Building_the_code)

You can purchase GstPreRecord product at:

[RidgeRun's store](https://shop.ridgerun.com/products/gstreamer-pre-record-element-1)