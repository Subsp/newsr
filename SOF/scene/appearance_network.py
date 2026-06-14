#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

# This code based on VastGaussian (https://arxiv.org/abs/2402.17427), and modified from GOF (https://github.com/autonomousvision/gaussian-opacity-fields)
# https://github.com/autonomousvision/gaussian-opacity-fields/blob/5245b20e5d11acd6d1ff5af4b890dc2bedd99693/scene/appearance_network.py#L5

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.loss_utils import l1_loss

class UpsampleBlock(nn.Module):
    def __init__(self, num_input_channels, num_output_channels):
        super(UpsampleBlock, self).__init__()
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv = nn.Conv2d(num_input_channels // (2 * 2), num_output_channels, 3, stride=1, padding=1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.pixel_shuffle(x)
        x = self.conv(x)
        x = self.relu(x)
        return x
    
class AppearanceNetwork(nn.Module):
    def __init__(self, num_input_channels, num_output_channels):
        super(AppearanceNetwork, self).__init__()
        
        self.conv1 = nn.Conv2d(num_input_channels, 256, 3, stride=1, padding=1)
        self.up1 = UpsampleBlock(256, 128)
        self.up2 = UpsampleBlock(128, 64)
        self.up3 = UpsampleBlock(64, 32)
        self.up4 = UpsampleBlock(32, 16)
        
        self.conv2 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, num_output_channels, 3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        # bilinear interpolation
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.sigmoid(x)
        return x


class AppearanceEmbedding():
    def __init__(self, num_views):
        
        STD = 1e-4
        
        self._appearance_embeddings = nn.Parameter(torch.empty(num_views, 64).cuda())
        self._appearance_embeddings.data.normal_(0, std=STD)
        
        self.appearance_network = AppearanceNetwork(3+64, 3).cuda()
        
    def capture(self):
        return (
            self._appearance_embeddings,
            self.appearance_network
        )
        
    def restore(self, appearance_embeddings, appearance_network):
        self._appearance_embeddings = appearance_embeddings
        self.appearance_network = appearance_network
        
    def get_apperance_embedding(self, idx):
        return self._appearance_embeddings[idx]
    
    def L1_loss_appearance(self, image, gt_image, view_idx):
        appearance_embedding = self.get_apperance_embedding(idx=view_idx)
        # center crop the image
        origH, origW = image.shape[1:]
        H = origH // 32 * 32
        W = origW // 32 * 32
        left = origW // 2 - W // 2
        top = origH // 2 - H // 2
        crop_image = image[:, top:top+H, left:left+W]
        crop_gt_image = gt_image[:, top:top+H, left:left+W]
        
        # down sample the image
        crop_image_down = torch.nn.functional.interpolate(crop_image[None], size=(H//32, W//32), mode="bilinear", align_corners=True)[0]
        
        crop_image_down = torch.cat([crop_image_down, appearance_embedding[None].repeat(H//32, W//32, 1).permute(2, 0, 1)], dim=0)[None]
        mapping_image = self.appearance_network(crop_image_down)
        transformed_image = mapping_image * crop_image
        
        return l1_loss(transformed_image, crop_gt_image)
        
# This code based on PGSR (https://arxiv.org/abs/2406.06521)

class PGSREmbedding():
    def __init__(self, num_views):
        
        self._As = torch.zeros((num_views)).cuda()
        self._Bs = torch.zeros((num_views)).cuda()
        
    def capture(self):
        raise NotImplementedError()
        
    def restore(self, appearance_embeddings, appearance_network):
        raise NotImplementedError()
    
    def get_apperance_embedding(self, idx):
        return self._As[idx], self._Bs[idx]
    
    def L1_loss_appearance(self, image, gt_image, view_idx):
        a, b = self.get_apperance_embedding(view_idx)
        transformed_image = torch.exp(a) * image + b
        
        return l1_loss(transformed_image, gt_image)