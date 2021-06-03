#!/bin/bash

# Downlaod models
mkdir models
cd models
wget "https://www.dropbox.com/s/yus207vi8t83d6z/segmentation_bike.pt" -q -O segmentation_bike.pt
wget "https://www.dropbox.com/s/uxgelj4f50hqq2h/stylegan2_bike.pt" -q -O stylegan2_bike.pt
cd ..

# Download dataset
mkdir datasets
cd datasets
wget "https://www.dropbox.com/s/0ybsudabqscstf7/biked_dataset.tar.gz" -q -O biked_dataset.tar.gz
tar -zxvf biked_dataset.tar.gz
rm biked_dataset.tar.gz

wget "https://www.dropbox.com/s/p6a615wd8qh6j7h/test_data.tar.gz" -q -O test_data.tar.gz
tar -zxvf test_data.tar.gz
rm test_data.tar.gz