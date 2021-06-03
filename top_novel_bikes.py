from utils import zdataset, show, labwidget, renormalize
from rewrite import ganrewrite, rewriteapp
import torch, copy, os, json
from torchvision.utils import save_image
from torchvision import transforms
import utils.stylegan2, utils.proggan
from utils.stylegan2 import load_seq_stylegan

import numpy as np
from sklearn.metrics import jaccard_score
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm, trange
import cv2

from utils import unet, anomaly

model_path = "./models/stylegan2_bike.pt"
model_size = 512
truncation = 0.5

name='bike'

seg_model_path = './models/segmentation_bike.pt'

data_path = './datasets/biked'
# Using full bike dataset will crash due to RAM limit on Colab. Please use appropriate hardware or reduce the dataset size. 
# The dataset used will affect the novelty detection result.
# This demo uses the test dataset containing 50 images as opposed to full dataset with ~4k images, the result is different with the paper

k=50
anomaly_threshold = 3.5

# Copy id for frame example shown in overleaf paper
# 907, 728, 348, 960

# Copy id for handle example shown in overleaf paper
# 580, 811, 576

copy_id=907
paste_id=7
key_ids=list(range(7,7+5))

seg_class = 7
channels=[0, 3]
# 0 - frame
# 1 - saddle
# 2 - wheel
# 3 - handle
# eg. [0, 3] - only frame or handle will be used for rewriting

layer=6
rank=30
lr=0.05
niter=2000

use_copy_as_paste_mask = False
dilate_mask= True
dilate_kernel_size=(16,16)


# Choices: ganname = 'stylegan' or ganname = 'proggan'
ganname = 'stylegan'

modelname = name

layernum = layer

# Number of images to sample when gathering statistics.
size = 10000

# Make a directory for caching some data.
layerscheme = 'default'
expdir = 'results/pgw/%s/%s/%s/layer%d' % (ganname, modelname, layerscheme, layernum)
os.makedirs(expdir, exist_ok=True)

# Load (and download) a pretrained GAN
if ganname == 'stylegan':
    model = load_seq_stylegan(model_path, path=True, size=model_size, mconv='seq', truncation=truncation)
    Rewriter = ganrewrite.SeqStyleGanRewriter
elif ganname == 'proggan':
    model = utils.proggan.load_pretrained(modelname)
    Rewriter = ganrewrite.ProgressiveGanRewriter
    
# Create a Rewriter object - this implements our method.
zds = zdataset.z_dataset_for_model(model, size=size)
gw = Rewriter(
    model, zds, layernum, cachedir=expdir,
    low_rank_insert=True, low_rank_gradient=False,
    use_linear_insert=False,  # If you set this to True, increase learning rate.e
    key_method='zca')

# Display a user interface to allow model rewriting.
savedir = f'masks/{ganname}/{modelname}'
interface = rewriteapp.GanRewriteApp(gw, size=256, mask_dir=savedir, num_canvases=32)

# Create detector instance given a directory of the normal images
ad = anomaly.AnomalyDetector(data_path, name=name, topk=k)

# Extract and cache embeddings of the normal images
ad.load_train_features()


import matplotlib.pyplot as plt

anomaly_scores = []
for i in range(0, 1000, 100):
    images = gw.render_image_batch(list(range(i,i+100)))
    scores = ad.predict_anomaly_scores(images)
    anomaly_scores.append(scores)

anomaly_scores = np.concatenate(anomaly_scores)
top_idx = anomaly_scores.argsort()[::-1]

row = 4
col = 5

fig, axes = plt.subplots(row, col, figsize=(col*4,row*3))

for i in range(row*col):
    image = gw.render_image(top_idx[i])
    ax = axes[i//col, i%col]
    ax.imshow(image)
    ax.title.set_text(top_idx[i])
    ax.axis('off')

fig.tight_layout(pad=0)
fig.savefig(f'bike_gan_novel_top{row*col}.jpg', bbox_inches='tight')
# fig.show()

import numpy as np
from tqdm import tqdm
from PIL import Image

train_dataset = anomaly.NormalDataset(data_path, grayscale=False, normalize=False)
image_files = train_dataset.x

anomaly_scores = []
for i in range(0, 1000, 100):
    images = []
    for x in range(i, i+100):
        images.append(Image.open(image_files[x]).resize((256,256)))
    scores = ad.predict_anomaly_scores(images, topk=k)
    anomaly_scores.append(scores)

anomaly_scores = np.concatenate(anomaly_scores)
top_idx = anomaly_scores.argsort()[::-1]

row = 4
col = 5

fig, axes = plt.subplots(row, col, figsize=(col*4,row*3))

for i in range(row*col):
    image = Image.open(image_files[top_idx[i]]).resize((256,256))
    ax = axes[i//col, i%col]
    ax.imshow(image)
    ax.title.set_text(top_idx[i])
    ax.axis('off')

fig.tight_layout(pad=0)
fig.savefig(f'bike_dataset_novel_top{row*col}.jpg', bbox_inches='tight')