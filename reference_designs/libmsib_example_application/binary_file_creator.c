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

#include <stdio.h>
#include <stdlib.h>

void printUsage() {
    printf("Usage: ./decoder -n <name_of_file> decimal_values...\n");
}

int main(int argc, char *argv[]) {
    if (argc < 4 || strcmp(argv[1], "-n") != 0) {
        // Incorrect number of arguments or missing -n flag
        printUsage();
        return 1;
    }

    char *outputFileName = argv[2];
    FILE *outputFile = fopen(outputFileName, "wb");

    if (outputFile == NULL) {
        perror("Error opening output file");
        return 1;
    }

    // Parse and write decimal values to binary file
    for (int i = 3; i < argc; i++) {
        int decimalValue = atoi(argv[i]);
        fputc(decimalValue, outputFile);
    }

    fclose(outputFile);

    printf("Conversion successful. Output file: %s\n", outputFileName);

    return 0;
}