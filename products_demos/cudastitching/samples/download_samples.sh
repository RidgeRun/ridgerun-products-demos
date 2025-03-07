#!/bin/bash

# Download script for sample images
export FILEID1=1870uf2hboMTYlLN8pqt5TZ2vrmPmWAjw
export FILENAME1=Park_30s.tar.bz
export FILEID2=1zCMjiOjNaGuH4mtD8YMoDHUISiIiiHVe
export FILENAME2=Mosaic_1min.tar.bz2

echo "Downloading sample images ..."
pip install gdown
gdown "https://drive.google.com/uc?id=$FILEID1"
gdown "https://drive.google.com/uc?id=$FILEID2"


echo "Unpacking sample images..."

tar -xvjf $FILENAME1
tar -xvjf $FILENAME2

rm $FILENAME1
rm $FILENAME2

