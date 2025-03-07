#!/bin/bash

# Download script for sample video and gyro data

echo "Downloading sample images ..."
wget "https://drive.usercontent.google.com/download?id=1uW2sg3E2W2UOF9rjDaHG8eJus7m531_4&export=download&authuser=0&confirm=f" -O non_stabilized.mp4

wget "https://drive.usercontent.google.com/download?id=1LXjqut2c8YIiJg66vH_UyYSdP664OErw&export=download&authuser=0&confirm=f" -O raw_gyro_data.csv

