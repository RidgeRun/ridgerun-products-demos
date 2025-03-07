/* Copyright (C) 2024 RidgeRun, LLC (http://www.ridgerun.com)
 * All Rights Reserved.
 *
 * The contents of this software are proprietary and confidential to RidgeRun,
 * LLC.  No part of this program may be photocopied, reproduced or translated
 * into another programming language without prior written consent of
 * RidgeRun, LLC.  The user is free to modify the source code after obtaining
 * a software license from RidgeRun.  All source code changes must be provided
 * back to RidgeRun without any encumbrance.
 */

#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <libmisb/formatter/jsonformatter.hpp>
#include <libmisb/libmisb.hpp>
#include <libmisb/logger.hpp>
#include <libmisb/metadata.hpp>
#include <map>
#include <signal.h>
#include <string>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <thread>
#include <time.h>
#include <unistd.h>

static constexpr int WAIT_MILLIS_PRODUCER = 1000;
static constexpr int WAIT_MILLIS_CONSUMER = 500;

unsigned char buffer[1024];
bool terminate = false;

void signal_handler(int signum)
{
    if (signum == SIGINT) {
        terminate = true;
    }
}

void run_consumer(int fd);
void run_producer(int fd);

int main (int argc , char **argv)
{
    bool producer = false;
    int opt = -1;

    signal(SIGINT, signal_handler);

    // add getopts to set consumer or producer mode
    while ((opt = getopt(argc, argv, "p")) != -1) {
        switch (opt) {
        case 'p':
            producer = true;
            break;
        default:
            std::cerr << "Usage: " << argv[0] << " [-p]" << std::endl;
            return 1;
        }
    }

    //create a FIFO file to push the packet
    std::string fifo_file = "/tmp/metadata_fifo";
    mkfifo(fifo_file.c_str(), 0644);
    int fd = open(fifo_file.c_str(), O_RDWR | O_NONBLOCK);

    if (fd == -1) {
        std::cerr << "Failed to open FIFO file" << std::endl;
        return 1;
    }

    if (producer) {
        std::cout << "Producer mode" << std::endl;
        run_producer(fd);
    } else {
        std::cout << "Consumer mode" << std::endl;
        run_consumer(fd);
    }

    std::cout << "Exiting..." << std::endl;
    close(fd);

    return 0;
}

void run_producer(int fd) {
    libmisb::LibMisb libmisb;
    libmisb.SetDebuggerLever(LogLevel::LIBMISB_DEBUG);

    std::shared_ptr<libmisb::formatter::iFormatter> json_formatter =
    std::make_shared<libmisb::formatter::JsonFormatter>();
    libmisb.SetFormatter(json_formatter);

    std::string misb0601_key = "060E2B34020B01010E01030101000000";

    libmisb::Metadata metadata;
    metadata.SetKey(misb0601_key);

    std::string timestamp = "";
    char buf[30];
    time_t now = time(0);
    struct tm tstruct;
    struct timeval tv;


    while (!terminate)
    {
        timestamp = "";
        //fill buf with zeros
        memset(buf, 0, sizeof(buf));

        // get timestamp in format Oct. 24, 2008. 00:13:29.913
        now = time(0);
        tstruct = *localtime(&now); 
        gettimeofday(&tv, NULL);
        strftime(buf, sizeof(buf), "%b. %d, %Y. %H:%M:%S", &tstruct);
        timestamp = buf;

        //milliseconds
        timestamp += "." + std::to_string(tv.tv_usec / 1000);


        std::cout<< "Timestamp: " << timestamp << std::endl;

        std::vector<metadata_item> metadata_subitems = {
            {"2", timestamp, false, {}},
            {"3", "MISSION01", false, {}},
            {"4", "AF-102", false, {}},
            {"5", "159.97436", false, {}},
            {"15", "14190.7195", false, {}},
        };

        metadata.SetItems(metadata_subitems);

        std::vector <unsigned char> packet;
        int result_encode = libmisb.Encode(metadata, packet);
        if (result_encode) {
            std::cout << "Failed to encode metadata" << std::endl;
            return;
        }

        write(fd, packet.data(), packet.size());

        // print the encoded data
        std::cout << "Encoded data: " << std::endl;
        for (auto p : packet) {
            std::cout << std::hex << (int)p << " ";
        }
        std::cout << std::endl << std::endl;

        std::this_thread::sleep_for(std::chrono::milliseconds(WAIT_MILLIS_PRODUCER));
    }

}

void run_consumer(int fd) {
    // consumer mode
    // read the packet from the FIFO file
    libmisb::LibMisb libmisb;
    libmisb.SetDebuggerLever(LogLevel::LIBMISB_DEBUG);

    std::shared_ptr<libmisb::formatter::iFormatter> json_formatter =
    std::make_shared<libmisb::formatter::JsonFormatter>();
    libmisb.SetFormatter(json_formatter);

    std::vector <unsigned char> packet_read;
    while (!terminate){
        int bytes_read = read(fd, buffer, sizeof(buffer));

        if (bytes_read == -1) {
            // print dots to show that the program is still running
            std::cout << "." << std::flush;
            std::this_thread::sleep_for(std::chrono::milliseconds(WAIT_MILLIS_CONSUMER));
            continue;
        }
        else if (bytes_read == 0 ) {
            continue;
        }
        packet_read.insert(packet_read.end(), buffer, buffer + bytes_read);

        std::string output = libmisb.Decode(packet_read);
        packet_read.clear();
        std::cout << std::endl << "Decoded metadata:" << std::endl << output << std::endl;

    }
}
