/*
 * Copyright (C) 2024 RidgeRun, LLC (http://www.ridgerun.com)
 * All Rights Reserved.
 *
 * The contents of this software are proprietary and confidential to RidgeRun,
 * LLC.  No part of this program may be photocopied, reproduced, or translated
 * into another programming language without the prior written consent of 
 * RidgeRun, LLC.  The user is free to modify the source code after obtaining
 * a software license from RidgeRun.  All source code changes must be provided
 * back to RidgeRun without any encumbrance.
 */

#include <glib.h>
#include <glib-unix.h>
#include <gst/gst.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define PIPELINE                \
  "videotestsrc is-live=true ! x264enc ! " \
  "seiinject name=inject ! rtph264pay ! " \
  "application/x-rtp,media=video,clock-rate=90000,encoding-name=H264 ! " \
  "udpsink host=127.0.0.1 port=5555"


typedef struct _GstMetaDemo GstMetaDemo;
struct _GstMetaDemo {
  GstElement *pipeline;
  GstElement *seiinject;
  GMainLoop *main_loop;
  guint bus_watch_id;
};

static gboolean create_pipeline(GstMetaDemo *metademo);
static void start_pipeline(GstMetaDemo *metademo);
static void stop_pipeline(GstMetaDemo *metademo);
static void release_resources(GstMetaDemo *metademo);
static gboolean bus_call(GstBus *bus, GstMessage *msg, gpointer data);
static gboolean signal_handler(gpointer user_data);

int main(int argc, char *argv[]) {

  GstMetaDemo *metademo = g_malloc(sizeof(GstMetaDemo));
  if(!metademo){
    g_print("Could not create demo\n");
    return 1;
  }

  /* Initialization */
  gst_init(&argc, &argv);

  if (!create_pipeline(metademo)) {
    g_free(metademo);
    return 1;
  }

  /* Set the pipeline to "playing" state*/
  g_print("Playing pipeline\n");
  start_pipeline(metademo);

  /* Iterate */
  g_print("Running...\n");
  g_main_loop_run(metademo->main_loop);

  /* Out of the main loop, clean up nicely */
  g_print("Returned, stopping playback\n");
  release_resources(metademo);
  
  g_free(metademo);
  return 0;
}

static GstPadProbeReturn
cb_have_data (GstPad          *pad,
              GstPadProbeInfo *info,
              gpointer         user_data)
{
  GDateTime *date = NULL;
  gchar *date_str = NULL;
  
  /* Format date to be used as metadata */
  date = g_date_time_new_now_local ();
  date_str = g_date_time_format (date, "%a %b %e %H:%M:%S %Y");
  g_date_time_unref (date);

  /* Set metadata in buffer */
  g_object_set(user_data, "metadata", date_str, NULL);
  g_free (date_str);

  return GST_PAD_PROBE_OK;
}

static gboolean bus_call(GstBus *bus, GstMessage *msg, gpointer data) {
  GMainLoop *loop = (GMainLoop *)data;

  switch (GST_MESSAGE_TYPE(msg)) {

  case GST_MESSAGE_EOS:
    g_print("End of stream\n");
    g_main_loop_quit(loop);
    break;

  case GST_MESSAGE_ERROR: {
    gchar *debug;
    GError *error;

    gst_message_parse_error(msg, &error, &debug);
    g_free(debug);

    g_printerr("Error: %s\n", error->message);
    g_error_free(error);

    g_main_loop_quit(loop);
    break;
  }
  default:
    break;
  }

  return TRUE;
}

static gboolean signal_handler (gpointer user_data)
{
    GMainLoop * loop = (GMainLoop *)user_data;

    g_print ("Interrupt received, closing...\n");
    g_main_loop_quit (loop)  ;

    return TRUE;
}

static gboolean create_pipeline(GstMetaDemo *metademo) {

  GMainLoop *loop;

  GstElement *pipeline = NULL;
  GstElement *seiinject = NULL;
  GstBus *bus = NULL;
  GstPad *pad = NULL;
  GError *error = NULL;

  if (!metademo) {
    return FALSE;
  }

  loop = g_main_loop_new(NULL, FALSE);
  g_unix_signal_add (SIGINT, signal_handler, loop);

  pipeline = gst_parse_launch(PIPELINE, &error);

  if (error) {
    g_printerr("Unable to build pipeline (%s)\n", error->message);
    g_clear_error(&error);
    return FALSE;
  }

  seiinject = gst_bin_get_by_name(GST_BIN(pipeline), "inject");
  if (!seiinject) {
    g_printerr("Could not get metadata element\n");
    return FALSE;
  }

  /* Add pad probe */
  pad = gst_element_get_static_pad (seiinject, "src");
  if (!pad) {
    g_printerr("Could not get pad\n");
    return FALSE;
  }
  /* Callback is used to set updating metadata */
  gst_pad_add_probe (pad, GST_PAD_PROBE_TYPE_BUFFER,
      (GstPadProbeCallback) cb_have_data, seiinject, NULL);
  gst_object_unref (pad);

  /* we add a message handler */
  bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
  metademo->bus_watch_id = gst_bus_add_watch(bus, bus_call, loop);
  gst_object_unref(bus);

  metademo->pipeline = pipeline;
  metademo->main_loop = loop;
  metademo->seiinject = seiinject;

  return TRUE;
}

static void start_pipeline(GstMetaDemo *metademo) {
  gst_element_set_state(metademo->pipeline, GST_STATE_PLAYING);
}
static void stop_pipeline(GstMetaDemo *metademo) {
  gst_element_set_state(metademo->pipeline, GST_STATE_NULL);
}

static void release_resources(GstMetaDemo *metademo) {
  if (!metademo) {
    return;
  }

  stop_pipeline(metademo);
  
  if (metademo->pipeline) {
    gst_object_unref(metademo->pipeline);
    metademo->pipeline = NULL;
  }

  if (metademo->seiinject) {
    metademo->seiinject = NULL;
  }

  g_source_remove(metademo->bus_watch_id);

  if (metademo->main_loop) {
    g_main_loop_unref(metademo->main_loop);
    metademo->main_loop = NULL;
  }

}
