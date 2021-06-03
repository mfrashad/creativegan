import argparse
import numpy as np
import os
import pickle
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import wide_resnet50_2

from torchvision import transforms as T

from torch.utils.tensorboard import SummaryWriter

import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

class NormalDataset(Dataset):
    def __init__(self, path, resize=224, cropsize=224, grayscale=True, normalize=True, n=None):
        # assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)
        self.path = path
        self.resize = resize
        self.cropsize = cropsize
        
        # load dataset
        self.x = self.load_dataset_folder(n)

        # set transforms
        transform_x = [T.Resize(resize, Image.ANTIALIAS),
                                      T.CenterCrop(cropsize)]

        if grayscale:
              transform_x.append(T.Grayscale(num_output_channels=3))

        transform_x.append(T.ToTensor())
                                            
        if normalize:
            transform_x.append(T.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225]))
        self.transform_x = T.Compose(transform_x)

        

    def __getitem__(self, idx):
        x = self.x[idx]

        x = Image.open(x).convert('RGB')
        x = self.transform_x(x)
        return x

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self, n):
        img_dir = self.path
        x = []
        img_fpath_list = sorted([os.path.join(img_dir, f)
                                  for f in os.listdir(img_dir)
                                  if (f.endswith('.png') or f.endswith('.jpg'))])
        x.extend(img_fpath_list)

        return list(x[:n]) if n != None else list(x)

class InferenceDataset(Dataset):
    def __init__(self, images, resize=224, cropsize=224, grayscale=True, normalize=True):
        # assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)
        self.resize = resize
        self.cropsize = cropsize

        # load dataset
        self.images = images

        # set transforms
        transform_x = [T.Resize(resize, Image.ANTIALIAS),
                                      T.CenterCrop(cropsize)]

        if grayscale:
              transform_x.append(T.Grayscale(num_output_channels=3))

        transform_x.append(T.ToTensor())
                                            
        if normalize:
            transform_x.append(T.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225]))
        self.transform_x = T.Compose(transform_x)

    def __getitem__(self, idx):
        x = self.images[idx]
        x = self.transform_x(x)
        return x

    def __len__(self):
        return len(self.images)

class AnomalyDetector(object):
    def __init__(self, data_path, cache_path='./anomaly_cache', topk=5, resize=224, cropsize=224, grayscale=True, name='default'):
        self.name = name
        self.topk = topk
        self.data_path = data_path
        self.cache_path = cache_path
        self.resize = resize
        self.cropsize = cropsize
        self.grayscale = grayscale

        self.transform_x = T.Compose([T.Resize(resize, Image.ANTIALIAS),
                                      T.CenterCrop(cropsize),
                                      T.Grayscale(num_output_channels=3),
                                      T.ToTensor(),
                                      T.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])])
        
        self.train_dataset = NormalDataset(self.data_path, grayscale=self.grayscale, resize=self.resize, cropsize=self.cropsize)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=32, pin_memory=True)

        self.train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', []), ('avgpool', [])])
        self.test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', []), ('avgpool', [])])

        self.scores = []
        self.score_map_list = []

        self.topk_indexes = None

        # device setup
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # device = 'cpu'
        
        # load model
        self.model = wide_resnet50_2(pretrained=True, progress=True)
        self.model.to(device)
        self.model.eval()

        # set model's intermediate outputs
        self.outputs = []
        
        def hook(module, input, output):
            self.outputs.append(output)

        self.model.layer1[-1].register_forward_hook(hook)
        self.model.layer2[-1].register_forward_hook(hook)
        self.model.layer3[-1].register_forward_hook(hook)
        self.model.avgpool.register_forward_hook(hook)

        os.makedirs(os.path.join(self.cache_path, 'temp'), exist_ok=True)
    
    

    def load_train_features(self):

        # extract train set features
        train_feature_filepath = os.path.join(self.cache_path, 'temp', 'train_%s.pkl' % os.path.basename(os.path.normpath(self.data_path)))
        if not os.path.exists(train_feature_filepath):
            for x in tqdm(self.train_dataloader, '| feature extraction | train |'):
                # model prediction
                with torch.no_grad():
                    pred = self.model(torch.unsqueeze(torch.mean(x.to(device),1),1).repeat(1,3, 1,1))
                    # get intermediate layer outputs
                    for k, v in zip(self.train_outputs.keys(), self.outputs):
                        self.train_outputs[k].append(v.cpu())
                    # initialize hook outputs
                    self.outputs = []
            for k, v in self.train_outputs.items():
                self.train_outputs[k] = torch.cat(v, 0).cpu()
            # save extracted feature
            with open(train_feature_filepath, 'wb') as f:
                pickle.dump(self.train_outputs, f, protocol=4)
        else:
            print('load train set feature from: %s' % train_feature_filepath)
            with open(train_feature_filepath, 'rb') as f:
                self.train_outputs = pickle.load(f)

    def extract_image_features(self, images):
        inference_dataset = InferenceDataset(images, grayscale=self.grayscale, resize=self.resize, cropsize=self.cropsize)
        inference_dataloader = DataLoader(inference_dataset, batch_size=32, pin_memory=True)
        
        self.test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', []), ('avgpool', [])])
        
        for x in tqdm(inference_dataloader, '| feature extraction |'):
            # self.test_imgs.extend(x.cpu().detach().numpy())
            # model prediction
            with torch.no_grad():
                pred = self.model(x.to(device))
            # get intermediate layer outputs
            for k, v in zip(self.test_outputs.keys(), self.outputs):
                self.test_outputs[k].append(v)
            # initialize hook outputs
            self.outputs = []
        for k, v in self.test_outputs.items():
            self.test_outputs[k] = torch.cat(v, 0)
    
    def predict_anomaly_scores(self, images, topk=None):
        if len(self.train_outputs['avgpool']) == 0:
            self.load_train_features()
            
        if topk==None:
            topk = self.topk

        self.extract_image_features(images)
        
        dist_matrix = calc_dist_matrix(torch.flatten(self.test_outputs['avgpool'], 1),
                               torch.flatten(self.train_outputs['avgpool'], 1))
        
        topk_values, self.topk_indexes = torch.topk(dist_matrix, k=topk, dim=1, largest=False)
        scores = torch.mean(topk_values, 1).cpu().detach().numpy()
        self.scores = scores
        return scores
    
    def predict_anomaly_masks(self, images, topk=None):
        self.predict_anomaly_scores(images, topk=topk)

        score_map_list = []

        for t_idx in tqdm(range(self.test_outputs['avgpool'].shape[0]), '| localization | test|'):
            score_maps = []
            for layer_name in ['layer1', 'layer2', 'layer3']:  # for each layer

                # construct a gallery of features at all pixel locations of the K nearest neighbors
                topk_feat_map = self.train_outputs[layer_name][self.topk_indexes[t_idx]].to(device)
                test_feat_map = self.test_outputs[layer_name][t_idx:t_idx + 1].to(device)
                feat_gallery = topk_feat_map.transpose(3, 1).flatten(0, 2).unsqueeze(-1).unsqueeze(-1)

                # calculate distance matrix
                dist_matrix_list = []
                for d_idx in range(feat_gallery.shape[0] // 100):
                    dist_matrix = torch.pairwise_distance(feat_gallery[d_idx * 100:d_idx * 100 + 100], test_feat_map)
                    dist_matrix_list.append(dist_matrix)
                dist_matrix = torch.cat(dist_matrix_list, 0)

                # k nearest features from the gallery (k=1)
                score_map = torch.min(dist_matrix, dim=0)[0]
                score_map = F.interpolate(score_map.unsqueeze(0).unsqueeze(0), size=224,
                                          mode='bilinear', align_corners=False)
                score_maps.append(score_map)

            # average distance between the features
            score_map = torch.mean(torch.cat(score_maps, 0), dim=0)

            # apply gaussian smoothing on the score map
            score_map = gaussian_filter(score_map.squeeze().cpu().detach().numpy(), sigma=4)
            score_map_list.append(score_map)
        
        self.score_map_list = score_map_list
        return score_map_list
    
    def predict_anomaly_masks_alt(self,images):

        grads = []
        def grad_hook(module, input, output):
            grads.append(output[0])
        
        self.model.layer4[-3].register_backward_hook(grad_hook)
        self.model.layer4[-2].register_backward_hook(grad_hook)
        
        inference_dataset = InferenceDataset(images, grayscale=self.grayscale, resize=self.resize, cropsize=self.cropsize)
        inference_dataloader = DataLoader(inference_dataset, batch_size=32, pin_memory=True)
        grad_mask = []
        for x in tqdm(inference_dataloader, '| feature extraction |'):
            # self.test_imgs.extend(x.cpu().detach().numpy())
            # model prediction
            self.test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', []), ('avgpool', [])])
            in_images = x.to(device)
            in_images.requires_grad = True
            pred = self.model(in_images)
            # get intermediate layer outputs
            for k, v in zip(self.test_outputs.keys(), self.outputs):
                self.test_outputs[k]= v
            
            dist_matrix = calc_dist_matrix(torch.flatten(self.test_outputs['avgpool'], 1),
                               torch.flatten(self.train_outputs['avgpool'], 1))
            topk_values, self.topk_indexes = torch.topk(dist_matrix, k=self.topk, dim=1, largest=False)
            scores = torch.log(torch.mean(topk_values, 1))
            scores.backward(gradient=torch.ones_like(scores))
            grad_mask.append(grads)
            # initialize hook outputs
            self.outputs = []
            grads = []
            
        masks = [[],[]]
        for grad_m in grad_mask:
            for i in range(2):
                masks[i].append(grad_m[i])
        for i in range(2):
            masks[i] = torch.cat(masks[i],0)
#             masks[i] = F.interpolate(masks[i], size=224,mode='bilinear', align_corners=False).detach().numpy()
        masks = torch.stack(masks,0)
        masks = np.squeeze(F.interpolate(torch.unsqueeze(torch.max(masks[0]+masks[1],1)[0],1),size=224,mode='bilinear',align_corners=False).cpu().detach().numpy())
        return masks
    
    def threshold_masks(self, masks, threshold=3):
        results = []
        for x in masks:
            mask = x.copy()
            mask[mask <= threshold] = 0
            mask[mask > threshold] = 1
            results.append(mask)
        return results
    
    def visualize_embedding(self, images, logdir='./embeddings', train=True, n=None, threshold=1, sprite_size=32, grayscale=False):
        train_dataset = NormalDataset(self.data_path, grayscale=grayscale, normalize=False)
        train_dataloader = DataLoader(train_dataset, batch_size=32, pin_memory=True)
        inference_dataset = InferenceDataset(images, grayscale=grayscale, normalize=False)
        inference_dataloader = DataLoader(inference_dataset, batch_size=32, pin_memory=True)
        
        if len(self.train_outputs['avgpool']) == 0:
            self.load_train_features()
        
        anomaly_scores = self.predict_anomaly_scores(images)
        anomaly_labels = anomaly_scores.copy()
        anomaly_labels[anomaly_scores > threshold] = 1
        anomaly_labels[anomaly_scores <= threshold] = 0
        anomaly_labels = anomaly_labels.astype('uint8')
        
        train_embs = torch.flatten(self.train_outputs['avgpool'], 1).cpu().detach()
        test_embs = torch.flatten(self.test_outputs['avgpool'], 1).cpu().detach()
        
        if n != None:
            train_embs = train_embs[:n]
            
        train_ids = [f'train_{i}' for i in range(len(train_embs))]
        test_ids = [f'test_{i}' for i in range(len(test_embs))]
        ids = []
        
        sprites = []
                
        if train:
            embs = torch.cat([train_embs, test_embs])
            train_scores = torch.zeros(train_embs.shape[0]).numpy().astype('float')
            train_labels = torch.zeros(train_embs.shape[0]).numpy().astype('uint8')
            anomaly_labels = np.concatenate((train_labels, anomaly_labels))
            anomaly_scores = np.concatenate((train_scores, anomaly_scores))
            
            train_ids.extend(test_ids)
            ids = train_ids
            
            for x in tqdm(train_dataloader, '| creating image sprites | train |'):
                x = F.interpolate(x, size=sprite_size)
                sprites.append(x)
                
                if n != None:
                    if len(sprites) * 32 > n:
                        sprites = [torch.cat(sprites)[:n]]
                        break
        else:
            embs = test_embs
            ids = test_ids
        
        for x in tqdm(inference_dataloader, '| creating image sprites | test |'):
            x = F.interpolate(x, size=sprite_size)
            sprites.append(x)
            
        sprites = torch.cat(sprites)
        

        metadata = [(ids[i], ids[i].split('_')[0], anomaly_labels[i], anomaly_scores[i]) for i in range(len(ids))]

        print("sprite shape", sprites.shape)
        print('embs shape', embs.shape)
        print("metadata len ", len(metadata))
        
        writer = SummaryWriter(log_dir=logdir)
        writer.add_embedding(embs, metadata=metadata, metadata_header=['id', 'dataset', 'novel', 'anomaly_score'], label_img=sprites)
        writer.close()
        
        print(f'run "%tensorboard --logdir {logdir}" to launch tensorboard')
        


def calc_dist_matrix(x, y):
    """Calculate Euclidean distance matrix with torch.tensor"""
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    x = x.unsqueeze(1).expand(n, m, d).to(device)
    y = y.unsqueeze(0).expand(n, m, d).to(device)
    dist_matrix = torch.sqrt(torch.pow(x - y, 2).sum(2))
    return dist_matrix
  
