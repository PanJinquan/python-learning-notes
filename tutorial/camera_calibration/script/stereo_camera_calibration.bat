#!/usr/bin/env bash
#set image_dir= data/bothImagesFixedStereo
#set width=9
#set height=6


set image_dir= data/data0516
set width= 8
set height=11


python core/stereo_camera_calibration.py ^
    --left_file config/left_cam.yml ^
    --right_file config/right_cam.yml ^
    --left_prefix left ^
    --right_prefix right ^
    --width %width% ^
    --height %height% ^
    --left_dir %image_dir% ^
    --right_dir %image_dir% ^
    --image_format png ^
    --square_size 0.020 ^
    --save_file config/stereo_cam.yml
