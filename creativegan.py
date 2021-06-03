import argparse
import re

from utils import zdataset, show, labwidget, renormalize
from rewrite import ganrewrite, rewriteapp
import torch, copy, os, json, shutil
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

from pytorch_msssim import ssim, ms_ssim

def _parse_num_range(s):
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]


parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, required=True, help='Name of experiment')
parser.add_argument('--model_path', type=str, required=True, help='Path to Stylegan model')
parser.add_argument('--model_size', type=int, default=512, help='GAN model output size')
parser.add_argument('--truncation', type=float, default=0.5, help="Value for truncation trick in Stylegan")

parser.add_argument('--seg_model_path', type=str, required=True, help="Path to segmentation model")
parser.add_argument('--seg_total_class', type=int, default=7, help="Total class/channel in segmentation model")
parser.add_argument('--seg_channels', type=_parse_num_range, required=True, help="List of segmentation channel that will be considered for rewriting")

parser.add_argument('--data_path', type=str, required=True, help='Path to dataset, the folder should directly contain the images')
parser.add_argument('--k', type=int, default=50, help='topk value for anomaly detection')
parser.add_argument('--anomaly_threshold', type=int, default=3.5, help='Threshold for novelty segmentation')

parser.add_argument('--copy_id', type=int, required=True, help='Seed id for target copy')
parser.add_argument('--paste_id', type=int, required=True, help='Seed id for target paste')
parser.add_argument('--context_ids', type=_parse_num_range, help='List of context ids', required=True)
parser.add_argument('--layernum', type=int, required=True, help='layer to be edited')
parser.add_argument('--rank', type=int, default=30, help='rank used in rewriting')
parser.add_argument('--lr', type=float, default=0.05, help='learning rate in rewriting')
parser.add_argument('--niter', type=float, default=2000, help='number of iterations in rewriting')

parser.add_argument('--n_outputs', type=int, default=9, help='Number of outputs to display')

parser.add_argument('--ssim', action='store_true', help="calculate ssim of modified model")
parser.add_argument('--novelty_score', action='store_true', help="calculate average novelty score of modified model")


args = parser.parse_args()

model_path = args.model_path
model_size = args.model_size
truncation = args.truncation

name=args.name

seg_model_path = args.seg_model_path

data_path = args.data_path
k=args.k
anomaly_threshold = args.anomaly_threshold
n_outputs = args.n_outputs

# Copy id for frame example shown in paper
# 907, 728, 348, 960

# Copy id for handle example shown in paper
# 580, 811, 576

copy_id=args.copy_id
paste_id=args.paste_id
key_ids=args.context_ids

seg_class = args.seg_total_class
channels=args.seg_channels
# 0 - frame
# 1 - saddle
# 2 - wheel
# 3 - handle
# eg. [0, 3] - only frame or handle will be used for rewriting

layer=args.layernum
rank=args.rank
lr=args.lr
niter=args.niter

use_copy_as_paste_mask = False
dilate_mask= True
dilate_kernel_size=(16,16)

def dilate(mask,kernel_size=(8,8)):
    kernel = np.ones(kernel_size, np.uint8)
    mask = cv2.dilate(mask, kernel)
    return mask

def segment(seg_model, images, ch=3, size=(224,224), threshold=0.5):
    trans = transforms.Compose([
                                transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                              ])
    images_tensor = torch.empty((len(images), ch, size[0], size[1]))
    for i in range(len(images)):
        images_tensor[i] = trans(images[i])
        
    seg_masks = seg_model(images_tensor.cuda()).sigmoid().detach().cpu()
    seg_masks = torch.where(seg_masks > threshold, torch.ones(seg_masks.size()), torch.zeros(seg_masks.size()))
    return seg_masks

def masks_to_colorimg(masks):
    colors = np.asarray([(201, 58, 64), (242, 207, 1), (0, 152, 75), (101, 172, 228),(56, 34, 132), (160, 194, 56), (155, 89, 182)])

    colorimg = np.ones((masks.shape[1], masks.shape[2], 3), dtype=np.float32) * 255
    channels, height, width = masks.shape

    for y in range(height):
        for x in range(width):
            selected_colors = colors[masks[:,y,x] > 0.5]

            if len(selected_colors) > 0:
                colorimg[y,x,:] = np.mean(selected_colors, axis=0)

    return colorimg.astype(np.uint8)



def find_best_seg_match(mask, seg_mask, channels=None):
    scores = []
    if channels is None:
        channels = list(range(seg_mask.shape[0]))
    for i in range(seg_mask.shape[0]):
        if i not in channels:
            scores.append(-1)
            continue
        
        iou_score = jaccard_score(mask.reshape(-1), seg_mask[i].reshape(-1))
        scores.append(iou_score)
    best_ch = np.argmax(scores)
    return best_ch

def render_mask(tup, gw, size=512):
    imgnum, mask = tup
    area = (renormalize.from_url(mask, target='pt', size=(size,size))[0] > 0.25)
    return gw.render_image(imgnum, mask=area)

def show_masks(masks, gw):
    n = len(masks)
    if n == 1:
        masks = masks[0]
    if type(masks) is tuple:
        plt.imshow(render_mask(masks, gw))
        return

    fig, axes = plt.subplots(1, n, figsize=(n*3, 3))
    for i in range(n):
        axes[i].imshow(render_mask(masks[i], gw))

if __name__ == '__main__':
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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seg_model = unet.ResNetUNet(seg_class).cuda()

    seg_model.load_state_dict(torch.load(seg_model_path))
    seg_model.eval()

    print('unet loaded')

    # Copy Mask
    image = gw.render_image(copy_id)
    copy_anomaly = ad.predict_anomaly_masks([image])
    copy_mask = ad.threshold_masks(copy_anomaly, threshold=anomaly_threshold)[0]
    seg_mask = segment(seg_model, [image])[0]

    best_ch = find_best_seg_match(copy_mask, seg_mask.numpy(), channels=channels)
    mask = seg_mask[best_ch].numpy()
    if dilate_mask:
        mask = dilate(mask, kernel_size=dilate_kernel_size)
    mask = Image.fromarray(mask.astype('uint8')*255, mode='L')
    area = (renormalize.from_image(mask, target='pt', size=(512,512))[0] > 0.25)
    mask_url = renormalize.as_url(mask)
    obj_acts, obj_output, obj_area, bounds = (gw.object_from_selection(copy_id, mask_url))
    interface.request['object'] = (copy_id, mask_url)


    # Paste Mask
    image = gw.render_image(paste_id)
    seg_mask = segment(seg_model, [image])[0]
    if not use_copy_as_paste_mask:
        mask = seg_mask[best_ch].numpy()
        if dilate_mask:
            mask = dilate(mask, kernel_size=dilate_kernel_size)
        mask = Image.fromarray(mask.astype('uint8')*255, mode='L')
    area = (renormalize.from_image(mask, target='pt', size=(512,512))[0] > 0.25)
    mask_url = renormalize.as_url(mask)
    interface.request['paste'] = (paste_id, mask_url)

    # Render Paste Image
    goal_in, goal_out, viz_out, bounds = gw.paste_from_selection(paste_id, mask_url, obj_acts, obj_area)
    imgout = renormalize.as_url(gw.render_object(viz_out, box=bounds))
    render_image = gw.render_object(viz_out, box=bounds)

    #Context Mask
    images = gw.render_image_batch(key_ids)
    seg_masks = segment(seg_model, images)

    best_seg_masks = seg_masks.permute(1,0,2,3)[best_ch]
    interface.request['key'] = []
    for i, idx in enumerate(key_ids):
        mask = best_seg_masks[i].numpy()
        if dilate_mask:
            mask = dilate(mask, kernel_size=dilate_kernel_size)
        mask = Image.fromarray(mask.astype('uint8')*255, mode='L')
        area = (renormalize.from_image(mask, target='pt', size=(512,512))[0] > 0.25)
        mask_url = renormalize.as_url(mask)
        interface.request['key'].append((idx, mask_url))

    # Rewriting
    def update_callback(it, loss):
        if it % 50 == 0 or it == niter - 1:
            loss_info = (f'lr {lr:.4f}\titer {it: 6d}/{niter: 6d}'
                        f'\tloss {loss.item():.4f}')
            print(loss_info, end='\r')
                    
    gw.apply_edit(interface.request,
                              rank=rank, niter=niter, piter=10, lr=lr,
                              update_callback=update_callback)


    imgnum, mask = interface.request['key'][0]
    key = gw.query_key_from_selection(imgnum, mask)
    sel, rq = gw.ranking_for_key(key, k=200)
    img_nums = sel.tolist()


    saved_state_dict = copy.deepcopy(gw.model.state_dict())

    with torch.no_grad():
        gw.model.load_state_dict(saved_state_dict)
        edited_images = gw.render_image_batch(img_nums)
        gw.model.load_state_dict(interface.original_model.state_dict())
        images = gw.render_image_batch(img_nums)

        
    # Visualize Result
    offset = 2
    n = n_outputs
    n_col = 3

    mask_savedir = "rewriting_masks"
    result_savedir = "rewriting_results"

    row = n//n_col
    col = n_col * 2
    fig, axes = plt.subplots(offset + row, col, figsize=(col*3, (offset+row)*3))

    for ax in axes.ravel():
        ax.axis('off')

    req = interface.request

    obj = render_mask(req['object'], gw)
    paste = render_mask(req['paste'], gw)
    axes[0, 0].imshow(obj)
    axes[0, 0].title.set_text('Copy')
    axes[0, 1].imshow(paste)
    axes[0, 1].title.set_text('Paste')
    axes[0, 2].imshow(render_image)

    axes[1, 0].title.set_text('Context')
    for i in range(min(n, len(req['key']))):
        context = render_mask(req['key'][i], gw)
        axes[1, i].imshow(context)


    for c in range(n_col):
        axes[offset, c*2].title.set_text('Original')
        axes[offset, c*2+1].title.set_text('Rewritten')

    for i in range(n):
        axes[offset+i%row, i//row*2].imshow(images[i])
        axes[offset+i%row, i//row*2 + 1].imshow(edited_images[i])
        
    fig.show()

    # Save Result
    os.makedirs(mask_savedir, exist_ok=True)
    os.makedirs(result_savedir, exist_ok=True)

    overwrite = False
    ver = 0
    name = f"{modelname}_c{copy_id}_p{paste_id}_layer{layernum}_rank{rank}_exp"

    while os.path.exists(os.path.join(result_savedir, name+str(ver)+'.png')) and not overwrite:
        ver += 1
    name = name + str(ver)
    fig.savefig(os.path.join(result_savedir, name), bbox_inches='tight')
    data = interface.request
    data['sel'] = img_nums

    def convert(o):
        if isinstance(o, np.int64): return int(o)  
        raise TypeError

    with open(os.path.join(mask_savedir, '%s.json' % name), 'w') as f:
        json.dump(data, f, indent=1, default=convert)

    
    if args.ssim:
        modified_dir = 'generated/modified'
        if os.path.exists(modified_dir):
            shutil.rmtree(modified_dir)
        os.makedirs(modified_dir)

        gw.model.load_state_dict(saved_state_dict)
        edited_images = gw.render_image_batch(list(range(100)))
        for i, im in enumerate(edited_images):
            filename = f'{i}.png'
            filename = os.path.join(modified_dir, filename)
            im.save(filename)
        
        from torch.utils.data import DataLoader
        
        # SSIM NOVELTY
        batch_size = 500
        gen_dataset1 = anomaly.NormalDataset(modified_dir, n=100, grayscale=False, normalize=False, resize=224, cropsize=224)
        gen_dataset2 = anomaly.NormalDataset(data_path, grayscale=False, normalize=False, resize=224, cropsize=224)
        gen_loader1 = DataLoader(gen_dataset1, batch_size=1, pin_memory=True)
        gen_loader2 = DataLoader(gen_dataset2, batch_size=batch_size, pin_memory=True)
        n = len(gen_dataset1)
        m = len(gen_dataset2)
        ssim_mat = torch.zeros((n, m))
    
        for i, xb in enumerate(gen_loader1):
            x = xb.repeat(batch_size, 1, 1, 1)
            for j, y in enumerate(gen_loader2):
                if len(y) < batch_size:
                    x = xb.repeat(len(y), 1, 1, 1)
                ssim_mat[i][j*batch_size:j*batch_size+len(y)] = (1 - ssim(x.cuda(), y.cuda(), data_range=1, size_average=False))/2

        # print('SSIM Mean: ', ssim_mat.mean().numpy())
        # print('SSIM Top 1 Mean: ', torch.topk(ssim_mat, k=1).values.mean().numpy())
        # print('SSIM Top 5 Mean: ', torch.topk(ssim_mat, k=5).values.mean().numpy())
        # print('SSIM Top 10 Mean: ', torch.topk(ssim_mat, k=10).values.mean().numpy())
        # print('SSIM Top 20 Mean: ', torch.topk(ssim_mat, k=20).values.mean().numpy())
        print('SSIM Top 50 Mean: ', torch.topk(ssim_mat, k=50).values.mean().numpy())

    if args.novelty_score:
        anomaly_scores = []
        for i in range(0, 1000, 100):
            images = gw.render_image_batch(list(range(i,i+100)))
            scores = ad.predict_anomaly_scores(images)
            anomaly_scores.append(scores)

        anomaly_scores = np.concatenate(anomaly_scores)
        print('Average Novelty Score', anomaly_scores.mean())

        
