#!/bin/bash

trap 'trap " " SIGTERM; kill 0; wait' SIGTERM
python spatial_sr_video.py --input_vid_path ./data/videos/bear_20_frames.mp4 --input_index 1 --batch_size 3 &
wait

trap 'trap " " SIGTERM; kill 0; wait' SIGTERM
python spatial_sr_video.py --input_vid_path ./data/videos/soupbox_20_frames.mp4 --input_index 1 --batch_size 3 &
wait

trap 'trap " " SIGTERM; kill 0; wait' SIGTERM
python spatial_sr_video.py --input_vid_path ./data/videos/car_turn_20_frames.mp4 --input_index 1 --batch_size 3 &
wait

trap 'trap " " SIGTERM; kill 0; wait' SIGTERM
python spatial_sr_video.py --input_vid_path ./data/videos/bike_picking_20_frames.mp4 --input_index 1 --batch_size 3 &
wait

trap 'trap " " SIGTERM; kill 0; wait' SIGTERM
python spatial_sr_video.py --input_vid_path ./data/videos/sheep_20_frames.mp4 --input_index 1 --batch_size 3 &
wait

trap 'trap " " SIGTERM; kill 0; wait' SIGTERM
python spatial_sr_video.py --input_vid_path ./data/videos/soccerball_20_frames.mp4 --input_index 1 --batch_size 3 &
wait

trap 'trap " " SIGTERM; kill 0; wait' SIGTERM
python spatial_sr_video.py --input_vid_path ./data/videos/camel_24_frames.mp4 --input_index 1 --batch_size 3 &
wait

trap 'trap " " SIGTERM; kill 0; wait' SIGTERM
python spatial_sr_video.py --input_vid_path ./data/videos/judo.mp4 --input_index 1 --batch_size 3 &
wait

trap 'trap " " SIGTERM; kill 0; wait' SIGTERM
python spatial_sr_video.py --input_vid_path ./data/videos/rollerblade.avi --input_index 1 --batch_size 3 &
wait

trap 'trap " " SIGTERM; kill 0; wait' SIGTERM
python spatial_sr_video.py --input_vid_path ./data/videos/dog.mp4 --input_index 1 --batch_size 3 &
wait


