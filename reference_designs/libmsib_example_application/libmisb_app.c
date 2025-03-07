/* Copyright (C) 2023 RidgeRun, LLC (http://www.ridgerun.com)
 * All Rights Reserved.
 *
 * The contents of this software are proprietary and confidential to RidgeRun,
 * LLC.  No part of this program may be photocopied, reproduced or translated
 * into another programming language without prior written consent of
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
#include <iostream>
#include <string>
#include <libmisb/libmisb.hpp>
#include "libmisb/formatter/jsonformatter.hpp"

#include <arpa/inet.h>

#define METADATA_PERIOD_SECS 1

static guint8 klv_metadata[61];

typedef struct _GstMetaDemo GstMetaDemo;
struct _GstMetaDemo {
  GstElement *pipeline;
  GstElement *metasrc;
  GstElement *filesink;
  GMainLoop *main_loop;
};

static gboolean create_pipeline(GstMetaDemo *metademo, const char *ip, const char *port, 
                                char *output_converter);
static void start_pipeline(GstMetaDemo *metademot);
static void stop_pipeline(GstMetaDemo *metademo);
static void release_resources(GstMetaDemo *metademo);
static gboolean bus_call(GstBus *bus, GstMessage *msg, gpointer data);
static gboolean handle_signal(gpointer data);

int is_valid_ip(const char *ip) {
    struct sockaddr_in sa;
    return inet_pton(AF_INET, ip, &(sa.sin_addr)) != 0;
}

int is_valid_port(const char *port) {
    char *endptr;
    long port_num = strtol(port, &endptr, 10);

    // Check for conversion errors or if the number is out of valid port range
    return *endptr == '\0' && port_num > 0 && port_num <= 65535;
}

std::string converter(const char *json_path){

  /* Initialize MISB object*/
  std::string input_file;
  libmisb::LibMisb libmisb;
  libmisb.SetDebuggerLever(LIBMISB_DEBUG);

  std::shared_ptr<libmisb::formatter::iFormatter> json_formatter =
    std::make_shared<libmisb::formatter::JsonFormatter>();
  libmisb.SetFormatter(json_formatter);

  input_file = json_path;
  
  std::ifstream file(input_file);
  std::string data;
  std::stringstream buffer;

  if (file.is_open()) {
    buffer << file.rdbuf();
    data = buffer.str();
    file.close();
  }

  std::vector<unsigned char> packet_encoded = libmisb.Encode(data);
  if (packet_encoded.empty()) {
    LOGGER.Log("Encode result packet is empty", LIBMISB_ERROR);
  }
  std::string string_packet = "";
  std::string string_byte;

    for (uint i = 0; i < packet_encoded.size(); i++) {
      string_byte = std::to_string(packet_encoded[i]);
      string_packet += string_byte;
      string_packet += " ";
    }
    LOGGER.Log(string_packet, LIBMISB_INFO);
  return string_packet;
}

int main(int argc, char *argv[]) {
  const char *ip = argv[1];
  const char *port = argv[2];
  const char *json_path = argv[3];

  printf("IP: %s\n", ip);
  printf("Port: %s\n", port);
  printf("JSON Path: %s\n", json_path);

  GstMetaDemo *metademo = static_cast<GstMetaDemo*>(g_malloc(sizeof(GstMetaDemo)));
  if(!metademo){
    g_print("Could not create demo\n");
    return 1;
  }

  g_unix_signal_add(SIGINT, (GSourceFunc)handle_signal, metademo);

  /* Initialization */
  gst_init(&argc, &argv);

  if (argc != 4) {
    fprintf(stderr, "Usage: %s <IP address> <Port number> <JSON file path>\n", argv[0]);
    return 1;
  }

  if (!is_valid_ip(ip) && !is_valid_port(port)) {
     g_print("Invalid IP address or port number!\n");
     g_print("Usage: %s <IP address> <Port number> <JSON file path>\n", argv[0]);
  }

  std::string string_converter = converter(json_path); 
  const int length = string_converter.length(); 
  char* output_converter = new char[length + 1]; 

  strcpy(output_converter, string_converter.c_str()); 
 
  if (!create_pipeline(metademo, ip, port, output_converter)) {
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

  return 0;
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

static gboolean create_pipeline(GstMetaDemo *metademo, const char *ip, const char *port, char *output_converter ) {

  GMainLoop *loop;

  GstElement *pipeline = NULL;
  GstElement *metasrc = NULL;
  GstElement *filesink = NULL;
  GstBus *bus = NULL;
  GByteArray *barray = NULL;
  guint8 *array_copy = NULL;
  guint metalen = 0;
  GError *error = NULL;

  if (!metademo) {
    return FALSE;
  }

  loop = g_main_loop_new(NULL, FALSE);

  char actual_pipeline[] = "metasrc period=1 metadata='%s' ! meta/x-klv ! mpegtsmux name=mux ! "
                      "udpsink host=%s port=%s videotestsrc is-live=true ! "
                      "video/x-raw,format=(string)I420,width=320,height=240,framerate=(fraction)30/1 ! x264enc ! mux.";

  char modified_pipeline[1024];

  // Use snprintf to format the string with the new values
  output_converter[strlen(output_converter) - 1] = '\0';
  snprintf(modified_pipeline, sizeof(modified_pipeline), actual_pipeline, output_converter, ip, port);

  // Now modified_pipeline contains the modified GStreamer pipeline
  pipeline = gst_parse_launch(modified_pipeline, &error);

  if (error) {
    g_printerr("Unable to build pipeline (%s)\n", error->message);
    g_clear_error(&error);
    return FALSE;
  }

  /*Prepare metadata*/

  /*We need to copy the array since the GByteArray will be the new owner and free it for us*/
  metalen = sizeof(klv_metadata);
  array_copy = static_cast<guint8*>(g_malloc(metalen));
  memcpy (array_copy, klv_metadata, metalen);
  
  barray = g_byte_array_new_take(array_copy, metalen);
  g_object_set(metasrc, "metadata-binary", barray, NULL);
  g_boxed_free(G_TYPE_BYTE_ARRAY, barray);

  g_object_set(G_OBJECT(metasrc), "period", METADATA_PERIOD_SECS, NULL);

  /* we add a message handler */
  bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
  gst_bus_add_watch(bus, bus_call, loop);
  gst_object_unref(bus);

  metademo->pipeline = pipeline;
  metademo->main_loop = loop;
  metademo->metasrc = metasrc;
  metademo->filesink = filesink;

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

  if (metademo->metasrc) {
    gst_object_unref(metademo->metasrc);
    metademo->metasrc = NULL;
  }

  if (metademo->filesink) {
    gst_object_unref(metademo->filesink);
    metademo->filesink = NULL;
  }

  if (metademo->main_loop) {
    g_main_loop_unref(metademo->main_loop);
    metademo->main_loop = NULL;
  }
}

static gboolean handle_signal(gpointer data) {
  GstMetaDemo *metademo = (GstMetaDemo *)data;

  g_main_loop_quit(metademo->main_loop);

  return TRUE;
}