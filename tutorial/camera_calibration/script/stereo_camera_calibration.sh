#!/usr/bin/env bash

image_dir=data/data0516
#image_dir=data/dataset
width=8
height=11
square_size=20 #mm
show=True

# left camera calibration
python core/single_camera_calibration.py \
    --image_dir  $image_dir \
    --image_format png  \
    --square_size $square_size  \
    --width $width  \
    --height $height  \
    --prefix left  \
    --save_file config/left_cam.yml \
    --show $show

# right camera calibration
python core/single_camera_calibration.py \
    --image_dir  $image_dir \
    --image_format png  \
    --square_size $square_size  \
    --width $width  \
    --height $height  \
    --prefix right  \
    --save_file config/right_cam.yml \
    --show $show

# stereo camera calibration
python core/stereo_camera_calibration.py \
    --left_file config/left_cam.yml \
    --right_file config/right_cam.yml \
    --left_prefix left \
    --right_prefix right \
    --width $width \
    --height $height \
    --left_dir $image_dir \
    --right_dir $image_dir \
    --image_format png \
    --square_size $square_size \
    --save_file config/stereo_cam.yml
