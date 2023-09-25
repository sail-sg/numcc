# Copyright 2023 Garena Online Private Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch.nn as nn
from src.fns import shrink_points_beyond_threshold, preprocess_img
from src.model.encoder import MCCEncoder
from src.model.decoder_anchor import DecoderPredictCenters
from src.model.decoder_feature import FeatureAggregator
import torch
import torch.nn.functional as F

class NUMCC(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, args=None):
        super().__init__()

        self.args = args
        self.encoder = MCCEncoder(args=args)
        self.decoderl1 = DecoderPredictCenters(args=args)
        self.decoderl2 = FeatureAggregator(nneigh=args.nneigh, args=args)

        self.fc_out = nn.Sequential(
            nn.ReLU(),
            nn.Linear(512, 1 + 256*3)
        )
        
    def forward(self, seen_images, seen_xyz, query_xyz, valid_seen_xyz, seen_xyz_hr=None, valid_seen_xyz_hr=None):

        query_xyz = shrink_points_beyond_threshold(query_xyz, self.args.shrink_threshold)

        seen_images_hr = None
        if seen_xyz_hr != None:
            seen_images_hr = preprocess_img(seen_images.clone(), res=self.args.xyz_size)
            seen_xyz_hr = shrink_points_beyond_threshold(seen_xyz_hr, self.args.shrink_threshold)

        seen_images = preprocess_img(seen_images)
        seen_xyz = shrink_points_beyond_threshold(seen_xyz, self.args.shrink_threshold)

        with torch.cuda.amp.autocast():
            latent, up_grid_fea = self.encoder(seen_images, seen_xyz, valid_seen_xyz, up_grid_bypass=seen_images_hr)
        fea = self.decoderl1(latent)

        if seen_xyz_hr == None:
            net = self.decoderl2(query_xyz, seen_xyz, valid_seen_xyz, fea, up_grid_fea)
        else:
            net = self.decoderl2(query_xyz, seen_xyz_hr, valid_seen_xyz_hr, fea, up_grid_fea)
            
        out = self.fc_out(net)

        return out, fea['anchors_xyz']

