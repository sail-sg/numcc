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

import numpy as np
import cv2
from tqdm import tqdm
import os
from pathlib import Path
import torch
from pytorch3d.io.obj_io import load_obj

import main_numcc
import util.misc as misc
from src.engine.engine import prepare_data_udf
from src.engine.engine_viz import generate_html_udf
from src.fns import *
from src.model.nu_mcc import NUMCC


def run_viz_udf(model, samples, device, args, prefix):
    model.eval()

    seen_xyz, valid_seen_xyz, query_xyz, unseen_rgb, labels, seen_images, gt_fps_xyz, seen_xyz_hr, valid_seen_xyz_hr = prepare_data_udf(samples, device, is_train=False, is_viz=True, args=args)
    seen_images_no_preprocess = seen_images.clone()


    with torch.no_grad():
        seen_images_hr = None
        
        if args.hr == 1:
            seen_images_hr = preprocess_img(seen_images.clone(), res=args.xyz_size)
            seen_xyz_hr = shrink_points_beyond_threshold(seen_xyz_hr, args.shrink_threshold)

        seen_images = preprocess_img(seen_images)
        query_xyz = shrink_points_beyond_threshold(query_xyz, args.shrink_threshold)
        seen_xyz = shrink_points_beyond_threshold(seen_xyz, args.shrink_threshold)


        latent, up_grid_fea = model.encoder(seen_images, seen_xyz, valid_seen_xyz, up_grid_bypass=seen_images_hr)
        fea = model.decoderl1(latent)
        centers_xyz = fea['anchors_xyz']
    

    max_n_queries_fwd = args.n_query_udf if not args.hr else int(args.n_query_udf * (args.xyz_size/args.xyz_size_hr)**2)

    # Filter query based on centers xyz # (1, 200, 3)
    offset = 0.3
    min_xyz = torch.min(centers_xyz, dim=1)[0][0] - offset
    max_xyz = torch.max(centers_xyz, dim=1)[0][0] + offset

    mask = (torch.rand(1, query_xyz.size()[1]) >= 0).to(args.device)
    mask = mask & (query_xyz[:,:,0] > min_xyz[0]) & (query_xyz[:,:,1] > min_xyz[1]) & (query_xyz[:,:,2] > min_xyz[2])
    mask = mask & (query_xyz[:,:,0] < max_xyz[0]) & (query_xyz[:,:,1] < max_xyz[1]) & (query_xyz[:,:,2] < max_xyz[2])
    query_xyz = query_xyz[mask].unsqueeze(0)

    total_n_passes = int(np.ceil(query_xyz.shape[1] / max_n_queries_fwd))

    pred_points = np.empty((0,3))
    pred_colors = np.empty((0,3))


    for param in model.parameters():
        param.requires_grad = False


    for p_idx in tqdm(range(total_n_passes)):
        p_start = p_idx     * max_n_queries_fwd
        p_end = (p_idx + 1) * max_n_queries_fwd
        cur_query_xyz = query_xyz[:, p_start:p_end]

        with torch.no_grad():
            if args.hr != 1:
                seen_points = seen_xyz
                valid_seen = valid_seen_xyz
            else:
                seen_points = seen_xyz_hr
                valid_seen = valid_seen_xyz_hr


            pred = model.decoderl2(cur_query_xyz, seen_points, valid_seen, fea, up_grid_fea, custom_centers = None)
            pred = model.fc_out(pred)

        max_dist = 0.5
        pred_udf = F.relu(pred[:,:,:1]).reshape((-1, 1)) # nQ, 1
        pred_udf = torch.clamp(pred_udf, max=max_dist) 

        # Candidate points
        t = args.udf_threshold
        pos = (pred_udf < t).squeeze(-1) # (nQ, )
        points = cur_query_xyz.squeeze(0) # (nQ, 3)
        points = points[pos].unsqueeze(0) # (1, n, 3)

        if torch.sum(pos) > 0:
            points = move_points(model, points, seen_points, valid_seen, fea, up_grid_fea, args, n_iter=args.udf_n_iter)

            # predict final color
            with torch.no_grad():
                pred = model.decoderl2(points, seen_points, valid_seen, fea, up_grid_fea)
                pred = model.fc_out(pred)

            cur_color_out = pred[:,:,1:].reshape((-1, 3, 256)).max(dim=2)[1] / 255.0
            cur_color_out = cur_color_out.detach().squeeze(0).cpu().numpy()
            if len(cur_color_out.shape) == 1:
                cur_color_out = cur_color_out[None,...]
            pts = points.detach().squeeze(0).cpu().numpy()
            pred_points = np.append(pred_points, pts, axis = 0)
            pred_colors = np.append(pred_colors, cur_color_out, axis = 0)
        
    img = (seen_images_no_preprocess[0].permute(1, 2, 0) * 255).cpu().numpy().copy().astype(np.uint8)

    fn_pc = None
    fn_pc_seen = None
    epoch = None
    if args.save_pc == 1:
        out_folder_ply = os.path.join('experiments/', f'{args.exp_name}', 'ply', 'epoch'+str(epoch).zfill(3))
        Path(out_folder_ply).mkdir(parents= True, exist_ok=True)
        prefix_pc = os.path.join(out_folder_ply, 'demo_udf')
        fn_pc = prefix_pc + '.ply'

        # seen
        out_folder_ply = os.path.join('experiments/', f'{args.exp_name}', 'ply_seen', 'epoch'+str(epoch).zfill(3))
        Path(out_folder_ply).mkdir(parents= True, exist_ok=True)
        prefix_pc = os.path.join(out_folder_ply, 'demo_udf')
        fn_pc_seen = prefix_pc +'_seen' +'.ply'

    with open(prefix + '.html', 'a') as f:
        generate_html_udf(
            img,
            seen_xyz, seen_images_no_preprocess,
            pred_points,
            pred_colors,
            query_xyz,
            f,
            centers = centers_xyz,
            fn_pc=fn_pc,
            fn_pc_seen = fn_pc_seen,
            pointcloud_marker_size=3
        )


def pad_image(im, value):
    if im.shape[0] > im.shape[1]:
        diff = im.shape[0] - im.shape[1]
        return torch.cat([im, (torch.zeros((im.shape[0], diff, im.shape[2])) + value)], dim=1)
    else:
        diff = im.shape[1] - im.shape[0]
        return torch.cat([im, (torch.zeros((diff, im.shape[1], im.shape[2])) + value)], dim=0)


def normalize(seen_xyz):
    seen_xyz = seen_xyz / (seen_xyz[torch.isfinite(seen_xyz.sum(dim=-1))].var(dim=0) ** 0.5).mean()
    seen_xyz = seen_xyz - seen_xyz[torch.isfinite(seen_xyz.sum(dim=-1))].mean(axis=0)
    return seen_xyz


def main_demo(args):

    misc.init_distributed_mode(args)

    model = NUMCC(args=args)
    model = model.to(args.device)
    misc.load_model(args=args, model_without_ddp=model, optimizer=None, loss_scaler=None)

    rgb = cv2.imread(args.image)
    obj = load_obj(args.point_cloud)

    seen_rgb = (torch.tensor(rgb).float() / 255)[..., [2, 1, 0]]

    H = 640
    W = 480
    seen_rgb = torch.nn.functional.interpolate(
        seen_rgb.permute(2, 0, 1)[None],
        size=[H, W],
        mode="bilinear",
        align_corners=False,
    )[0].permute(1, 2, 0)

    seen_xyz = obj[0].reshape(H, W, 3)
    seg = cv2.imread(args.seg, cv2.IMREAD_UNCHANGED)
    mask = torch.tensor(cv2.resize(seg, (W, H))).bool()
    seen_xyz[~mask] = float('inf')


    seen_xyz = normalize(seen_xyz)

    bottom, right = mask.nonzero().max(dim=0)[0]
    top, left = mask.nonzero().min(dim=0)[0]

    bottom, right = mask.nonzero().max(dim=0)[0]
    top, left = mask.nonzero().min(dim=0)[0]

    margin = 40
    bottom = bottom + margin
    right = right + margin
    top = max(top - margin, 0)
    left = max(left - margin, 0)

    seen_xyz = seen_xyz[top:bottom+1, left:right+1]
    seen_rgb = seen_rgb[top:bottom+1, left:right+1]

    seen_xyz = pad_image(seen_xyz, float('inf'))
    seen_rgb = pad_image(seen_rgb, 0)

    seen_rgb = torch.nn.functional.interpolate(
        seen_rgb.permute(2, 0, 1)[None],
        size=[800, 800],
        mode="bilinear",
        align_corners=False,
    )

    seen_xyz_ori = seen_xyz.clone()

    seen_xyz = torch.nn.functional.interpolate(
        seen_xyz.permute(2, 0, 1)[None],
        size=[112, 112],
        mode="bilinear",
        align_corners=False,
    ).permute(0, 2, 3, 1)

    seen_xyz_hr = torch.nn.functional.interpolate(
        seen_xyz_ori.permute(2, 0, 1)[None],
        size=[args.xyz_size, args.xyz_size],
        mode="bilinear",
        align_corners=False,
    ).permute(0, 2, 3, 1)

    samples = [
        [seen_xyz, seen_rgb, seen_xyz_hr],
        [torch.zeros((20000, 3)), torch.zeros((20000, 3))],
    ]
    run_viz_udf(model, samples, "cuda", args, prefix=args.output)


if __name__ == '__main__':
    parser = main_numcc.get_args_parser()
    parser.add_argument('--image', default='demo/iphone/luggage/im.jpg', type=str, help='input image file')
    parser.add_argument('--point_cloud', default='demo/iphone/luggage/pc.obj', type=str, help='input obj file')
    parser.add_argument('--seg', default='demo/iphone/luggage/seg.png', type=str, help='input segmentation file')
    parser.add_argument('--output', default='demo/output', type=str, help='output path')
    parser.add_argument('--checkpoint', default='pretrained/udf-ep99.pth', type=str, help='model checkpoint')
    parser.add_argument('--batch_query', default=48000, type=int, help='batch query for repulsive UDF')

    parser.set_defaults(eval=True)

    args = parser.parse_args()
    args.resume = args.checkpoint
    args.n_query_udf = args.batch_query
    main_demo(args)

