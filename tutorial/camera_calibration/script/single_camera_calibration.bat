set image_dir=data/data0516
set width=8
set height=11
set show=False


python core/single_camera_calibration.py ^
    --image_dir  %image_dir% ^
    --image_format png  ^
    --square_size 0.020  ^
    --width %width%  ^
    --height %height%  ^
    --prefix left  ^
    --save_file config/left_cam.yml ^
    --show %show%

python core/single_camera_calibration.py ^
    --image_dir  %image_dir% ^
    --image_format png  ^
    --square_size 0.020  ^
    --width %width%  ^
    --height %height%  ^
    --prefix right  ^
    --save_file config/right_cam.yml ^
    --show %show%
