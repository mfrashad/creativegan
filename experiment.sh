for frame in 907 728 348 960
do
	python creativegan.py --name "bike" \
                       --model_path "./models/stylegan2_bike.pt" \
                       --seg_model_path './models/segmentation_bike.pt' \
                       --seg_channels 0,3 \
                       --data_path './datasets/test_data' \
                       --copy_id $frame \
                       --paste_id 7 \
                       --context_ids 7-12 \
                       --layernum 6
done

for handle in 580 811 576
do
	python creativegan.py --name "bike" \
                       --model_path "./models/stylegan2_bike.pt" \
                       --seg_model_path './models/segmentation_bike.pt' \
                       --seg_channels 3 \
                       --data_path './datasets/test_data' \
                       --copy_id $handle \
                       --paste_id 7 \
                       --context_ids 7-12 \
                       --layernum 8
done