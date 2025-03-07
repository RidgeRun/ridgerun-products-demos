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

#define MAPPING1 "/stream1"
#define MAPPING2 "/stream2"
#define SERVICE "12345"
#define USER "anonymous"
#define PASSWD "secret"
#ifdef JETSON
#define ENC264 "nvvidconv ! nvv4l2h264enc iframeinterval=15 idrinterval=15 insert-sps-pps=true maxperf-enable=true"
#else
#define ENC264 "x264enc tune=zerolatency key-int-max=15"
#endif

gboolean
signal_handler (gpointer user_data)
{
    GMainLoop * loop = (GMainLoop *)user_data;

    g_print ("Interrupt received, closing...\n");
    g_main_loop_quit (loop)  ;

    return TRUE;
}

int main (gint argc, gchar *argv[])
{
    GstElement * pipeline;
    GMainLoop * loop;
    gint ret = -1;
    const gchar * default_desc;
    const gchar * second_desc;
    const gchar * final_desc;
    gboolean camera_stream = FALSE;
    GError *error = NULL;
    
    gst_init (&argc, &argv);

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            g_print ("-h \t--help: print this help text \n");
            g_print ("-c \t--camera: enable stream from camera \n");
            ret = 0;
            goto out;
        } else if (strcmp(argv[i], "-c") == 0 || strcmp(argv[i], "--camera") == 0) {
            camera_stream = TRUE;
        }
    }

    default_desc = "rtspsink name=sink service="SERVICE" auth="USER":"PASSWD" videotestsrc is-live=true ! "ENC264" ! "
                "h264parse ! video/x-h264,mapping="MAPPING1" ! sink. audiotestsrc ! queue ! audioconvert ! "
                "queue ! voaacenc ! aacparse ! audio/mpeg,mapping="MAPPING1" ! sink.";
    
    if (camera_stream) {
        second_desc = " v4l2src ! queue ! videoconvert ! queue ! "ENC264" ! "
                    "h264parse ! video/x-h264,mapping="MAPPING2" ! sink.";
        final_desc = g_strconcat(default_desc, second_desc, NULL);
        pipeline = gst_parse_launch_full (final_desc, NULL, GST_PARSE_FLAG_FATAL_ERRORS, &error);
    } else {
        pipeline = gst_parse_launch_full (default_desc, NULL, GST_PARSE_FLAG_FATAL_ERRORS, &error);
    }

    if (!pipeline || error) {
        g_printerr ("Unable to build pipeline: %s", error->message ? error->message : "(no debug)");
        goto out;
    }

    /* Playing the pipeline */
    gst_element_set_state (pipeline, GST_STATE_PLAYING);
    g_print ("New RTSP stream started at rtsp://127.0.0.1:%s%s \n", SERVICE, MAPPING1);
    if (camera_stream) {
        g_print ("New Webcam RTSP stream started at rtsp://127.0.0.1:%s%s \n", SERVICE, MAPPING2);
    }
  
    /* Block until CTRL+C is pressed */
    loop = g_main_loop_new (NULL, TRUE);
    g_unix_signal_add (SIGINT, signal_handler, loop);
    g_main_loop_run (loop);
    g_main_loop_unref (loop);
  
    /* NULL the pipeline */
    gst_element_set_state (pipeline, GST_STATE_NULL);
    gst_object_unref (pipeline);
  
    /* Successful run */
    g_print ("Successfully closed\n");
    ret = 0;

 out:
    gst_deinit ();
  
    return ret;
}
