#!/bin/bash

# Download script for sample images

export fileid=1yoA7tNjSNGpodlM6ern_8oQbIWK2wfa0
export filename=bev_6_cameras.zip

echo "Downloading sample images ..."
wget --no-check-certificate "https://docs.google.com/uc?export=download&id=${fileid}" -O ${filename}

echo "Unpacking sample images..."

unzip $filename

rm -f $filename
