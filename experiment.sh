#!/bin/bash
python top_novel_bikes.py

for frame in 907 728 348 960
do
	python creativegan.py --name "bike" \
                       --model_path "./models/stylegan2_bike.pt" \
                       --seg_model_path './models/segmentation_bike.pt' \
                       --seg_channels 0,3 \
                       --data_path './datasets/biked' \
                       --copy_id $frame \
                       --paste_id 7 \
                       --context_ids 7-12 \
                       --layernum 6 \
                       --ssim \
                       --novelty_score
done

for handle in 580 811 576
do
	python creativegan.py --name "bike" \
                       --model_path "./models/stylegan2_bike.pt" \
                       --seg_model_path './models/segmentation_bike.pt' \
                       --seg_channels 3 \
                       --data_path './datasets/biked' \
                       --copy_id $handle \
                       --paste_id 7 \
                       --context_ids 7-12 \
                       --layernum 8 \
                       --ssim \
                       --novelty_score
done