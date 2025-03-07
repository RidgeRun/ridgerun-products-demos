#!/bin/bash

# Download script for sample images
export FILEID=1IbgHfp6ODZYFfdRi1rx7LJDPB9MKB6ri
export FILENAME=Example1.tar.bz2

echo "Downloading sample images ..."
pip install gdown
gdown "https://drive.google.com/uc?id=$FILEID"


echo "Unpacking sample images..."

tar -xvjf $FILENAME

rm $FILENAME

