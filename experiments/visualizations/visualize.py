import os
import numpy as np

import torch

from modules import WeightNet
from utils.utils import load_network, deform_with_MLS, build_dataloader
from utils.utils import save_pts
from visualizations.create_visualization import create_basic_visualization, create_temp_visualization


def visualize(opt, save_subdir="visualize"):
    opt.batch_size = 1
    dataloader = build_dataloader(opt)

    net = WeightNet(opt=opt).to(opt.device) # expects (batch_size, 3)
    net.eval()
    load_network(net, opt.ckpt, opt.device)

    vis_output_dir = os.path.join(opt.log_dir, save_subdir)
    os.makedirs(vis_output_dir, exist_ok=True)

    cp_offsets = np.array(opt.cp_offsets, dtype=np.float32)
    cp_order = opt.cp_order
    
    with torch.no_grad():
        for _, data in enumerate(dataloader):
            # get data
            data["source_shape"]    = data["source_shape"].detach().to(opt.device) 
            data["source_cp"]       = data["source_cp"].detach().to(opt.device) 
            target_cp = data["source_cp"].clone()[0]

            # add CP to the shapes for our case
            source_shape = torch.cat((data["source_shape"], data["source_cp"]), dim=1)

            # run network
            weights = net(source_shape)
            weights = weights / opt.T
            weights = torch.softmax(weights, dim=-1)

            # Source
            s_fn = data['source_file'][0]
            s_shape = data["source_shape"][0].clone().cpu()
            if 'source_face' in data and data['source_face'] is not None:
                save_pts(os.path.join(vis_output_dir,"{}-Sa.obj".format(s_fn)), s_shape, colors=np.array([0.84, 0.37, 0]), faces=data['source_face'][0].cpu())
            else:
                save_pts(os.path.join(vis_output_dir,"{}-Sa.obj".format(s_fn)), s_shape, colors=np.array([0.84, 0.37, 0]))
            # Control points
            save_pts(os.path.join(vis_output_dir,"source_points.obj"), data["source_cp"][0].clone().cpu(), colors=np.array([0.8, 0, 0.8]))

            for idx in range(len(cp_order)):
                target_cp[cp_order[idx]] = target_cp[cp_order[idx]] + cp_offsets[cp_order[idx]]
                
                deformed_shape, _, _ = deform_with_MLS(data["source_cp"], target_cp.unsqueeze(0), source_shape, None, weights, deform_type=opt.MLS_deform)
                deformed_classic, _, _ = deform_with_MLS(data["source_cp"], target_cp.unsqueeze(0), source_shape, None, None, alpha=opt.MLS_alpha, deform_type=opt.MLS_deform)

                # Deformed
                if 'source_face' in data and data['source_face'] is not None:
                    save_pts(os.path.join(vis_output_dir,"{}-Sab_{}.obj".format(s_fn, idx)), deformed_shape[0].cpu(), colors=np.array([0, 0.45, 0.7]), faces=data['source_face'][0].cpu())
                    save_pts(os.path.join(vis_output_dir,"{}-Sab_CLASSIC_{}.obj".format(s_fn, idx)), deformed_classic[0].cpu(), colors=np.array([0.5, 0.7, 0.2]), faces=data['source_face'][0].cpu())
                else:
                    save_pts(os.path.join(vis_output_dir,"{}-Sab_{}.obj".format(s_fn, idx)), deformed_shape[0].cpu(), colors=np.array([0, 0.45, 0.7]))
                    save_pts(os.path.join(vis_output_dir,"{}-Sab_CLASSIC_{}.obj".format(s_fn, idx)), deformed_classic[0].cpu(), colors=np.array([0.5, 0.7, 0.2]))
                # Control points
                save_pts(os.path.join(vis_output_dir,"target_points_{}.obj".format(idx)), target_cp.clone().cpu(), colors=np.array([0.2, 0.2, 0.2]))

            # create visualization
            create_basic_visualization(opt, vis_output_dir + os.path.sep, shape=s_fn, cam_radius=opt.cam_radius, num_images=12, delete_images=True)

def temperature_visualization(opt, save_subdir="visualize_temp"):
    opt.batch_size = 1
    dataloader = build_dataloader(opt)

    net = WeightNet(opt=opt).to(opt.device) # expects (batch_size, 3)
    net.eval()
    load_network(net, opt.ckpt, opt.device)

    vis_output_dir = os.path.join(opt.log_dir, save_subdir)
    os.makedirs(vis_output_dir, exist_ok=True)

    cp_offsets = np.array(opt.cp_offsets, dtype=np.float32)
    cp_order = opt.cp_order
    opt.temps = np.array(opt.temps, dtype=np.float32)
    
    with torch.no_grad():
        for _, data in enumerate(dataloader):
            ############# get data ###########
            data["source_shape"]    = data["source_shape"].detach().to(opt.device) 
            data["source_cp"]       = data["source_cp"].detach().to(opt.device) 

            # add CP to the shapes for our case
            source_shape = torch.cat((data["source_shape"], data["source_cp"]), dim=1)

            ############# run network ###########
            weights = net(source_shape)

            # Source
            s_fn = data['source_file'][0]
            s_shape = data["source_shape"][0].clone().cpu()
            if 'source_face' in data and data['source_face'] is not None:
                save_pts(os.path.join(vis_output_dir,"{}-Sa.obj".format(s_fn)), s_shape, colors=np.array([0.8, 0, 0]), faces=data['source_face'][0].cpu())
            else:
                save_pts(os.path.join(vis_output_dir,"{}-Sa.obj".format(s_fn)), s_shape, colors=np.array([0.8, 0, 0]))
            # Control points
            save_pts(os.path.join(vis_output_dir,"source_points.obj"), data["source_cp"][0].clone().cpu(), colors=np.array([0.8, 0, 0.8]))

            ############# run MLS ##############
            for T in opt.temps:
                target_cp = data["source_cp"].clone()[0]
                curr_weights = weights / T
                curr_weights = torch.softmax(curr_weights, dim=-1)

                for idx in range(len(cp_order)):
                    target_cp[cp_order[idx]] = target_cp[cp_order[idx]] + cp_offsets[cp_order[idx]]
                    
                    deformed_shape, _, _ = deform_with_MLS(data["source_cp"], target_cp.unsqueeze(0), source_shape, None, curr_weights, deform_type=opt.MLS_deform)

                    # Deformed
                    if 'source_face' in data and data['source_face'] is not None:
                        save_pts(os.path.join(vis_output_dir,"{}-Sab_{}_T_{:.3f}.obj".format(s_fn, idx, T)), deformed_shape[0].cpu(), colors=np.array([0, 0, 0.8]), faces=data['source_face'][0].cpu())
                    else:
                        save_pts(os.path.join(vis_output_dir,"{}-Sab_{}_T_{:.3f}.obj".format(s_fn, idx, T)), deformed_shape[0].cpu(), colors=np.array([0, 0, 0.8]))
                    # Control points
                    save_pts(os.path.join(vis_output_dir,"target_points_{}.obj".format(idx)), target_cp.clone().cpu(), colors=np.array([0.8, 0.3, 0.8]))

            # create visualization
            create_temp_visualization(opt, vis_output_dir + os.path.sep, shape=s_fn, cam_radius=4.5, delete_images=True)