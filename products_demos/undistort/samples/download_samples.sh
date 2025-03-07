#!/bin/bash

# Download script for sample images
export FILEID2=108R0MPzmX3tefjZo-RwohGA6T6ozoC4O
export FILENAME2=undistort_brown_conrady_test.tar.bz2

echo "Downloading sample images ..."
pip install gdown
gdown "https://drive.google.com/uc?id=$FILEID2"

echo "Unpacking sample images..."

tar -xvf $FILENAME2

rm $FILENAME2

