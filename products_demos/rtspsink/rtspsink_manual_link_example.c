/* 
 * Copyright (C) 2016-2024 RidgeRun, LLC (http://www.ridgerun.com)
 * All Rights Reserved.
 *
 * The contents of this software are proprietary and confidential to RidgeRun,
 * LLC.  No part of this program may be photocopied, reproduced, or translated
 * into another programming language without the prior written consent of 
 * RidgeRun, LLC.  The user is free to modify the source code after obtaining
 * a software license from RidgeRun.  All source code changes must be provided
 * back to RidgeRun without any encumbrance.
 */

#include <gst/gst.h>
#include <glib-unix.h>

#define MAPPING "/ridgerun"
#define SERVICE "12345"
#ifdef JETSON
#define ENC264 "nvv4l2h264enc"
#define VIDCONV "nvvidconv"
#else
#define ENC264 "x264enc"
#define VIDCONV "identity"
#endif

gboolean
signal_handler (gpointer user_data)
{
    GMainLoop * loop = (GMainLoop *)user_data;

    g_print ("Interrupt received, closing...\n");
    g_main_loop_quit (loop);

    return TRUE;
}

int main (gint argc, gchar *argv[])
{
    GstElement * pipeline;
    GstElement * vts;
    GstElement * encoder;
    GstElement * converter;
    GstElement * capsfilter;
    GstElement * rtsp;
    GstCaps * caps;
    GMainLoop * loop;
    gint ret = -1;
  
    gst_init (&argc, &argv);

    pipeline = gst_pipeline_new ("GstRtspSink");

    /* Creating the different elements in the pipeline */
    vts = gst_element_factory_make ("videotestsrc", "videotestsrc");
    if (!vts) {
        g_printerr ("Unable to create videotestsrc\n");
        goto no_vts;
    }
    g_object_set (vts, "is-live", TRUE, NULL);
  
    encoder = gst_element_factory_make (ENC264, ENC264);
    if (!encoder) {
        g_printerr ("Unable to create "ENC264"\n");
        goto no_enc;
    }
#ifdef JETSON
    g_object_set (encoder, "iframeinterval", 15, "idrinterval", 15, "insert-sps-pps", TRUE, "maxperf-enable", TRUE, NULL);
#else
    g_object_set (encoder, "speed-preset", 1, "key-int-max", 15, NULL);
#endif

    capsfilter = gst_element_factory_make ("capsfilter", "capsfilter");
    if (!capsfilter) {
        g_printerr ("Unable to create capsfilter\n");
        goto no_caps;
    }
    caps = gst_caps_from_string ("video/x-h264,mapping=" MAPPING);
    g_object_set (capsfilter, "caps", caps, NULL);
    gst_caps_unref (caps);

    converter = gst_element_factory_make (VIDCONV,VIDCONV);
    if (!converter) {
        g_printerr ("Unable to create "VIDCONV"\n");
        goto no_conv;
    }

    rtsp = gst_element_factory_make ("rtspsink", "rtspsink");
    if (!rtsp) {
        g_printerr ("Unable to create rtspsink\n");
        goto no_rtsp;
    }
    g_object_set (rtsp, "service", SERVICE, NULL);

    /* Linking all the elements together */
    gst_bin_add_many (GST_BIN (pipeline), vts,converter,encoder,capsfilter, rtsp, NULL);
    gst_element_link_many (vts, encoder, capsfilter, rtsp, NULL);

    /* Playing the pipeline */
    gst_element_set_state (pipeline, GST_STATE_PLAYING);
    g_print ("New RTSP stream started at rtsp://127.0.0.1:" SERVICE MAPPING "\n");
  
    /* Block until CTRL+C is pressed */
    loop = g_main_loop_new (NULL, TRUE);
    g_unix_signal_add (SIGINT, signal_handler, loop);
    g_main_loop_run (loop);
    g_main_loop_unref (loop);
  
    /* NULL the pipeline */
    gst_element_set_state (pipeline, GST_STATE_NULL);

    /* Successful run */
    g_print ("Successfully closed\n");
    ret = 0;
    goto no_vts;
  
 no_rtsp:
    gst_object_unref (capsfilter);
  
 no_caps:
    gst_object_unref (encoder);
  
 no_enc:
    gst_object_unref (vts);

 no_conv:
    gst_object_unref (converter);
  
 no_vts:
    gst_object_unref (pipeline);
    gst_deinit ();
  
    return ret;
}
