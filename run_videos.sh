#!/bin/bash

python denoising_video.py --input_vid_path ./data/videos/bear_20_frames.mp4 --input_index 1 --batch_size 3
python denoising_video.py --input_vid_path ./data/videos/soupbox_20_frames.mp4 --input_index 1 --batch_size 3
python denoising_video.py --input_vid_path ./data/videos/car_turn_20_frames.mp4 --input_index 1 --batch_size 3
python denoising_video.py --input_vid_path ./data/videos/bike_picking_20_frames.mp4 --input_index 1 --batch_size 3