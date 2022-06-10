import os
import numpy as np
import torch

from utils.utils import deform_with_MLS, build_dataloader
from utils.utils import save_pts
from visualizations.create_visualization import create_temp_visualization


def mls_ablation(opt, save_subdir="mls_ablation"):
    opt.batch_size = 1
    dataloader = build_dataloader(opt)

    vis_output_dir = os.path.join(opt.log_dir, save_subdir)
    os.makedirs(vis_output_dir, exist_ok=True)

    cp_offsets = np.array(opt.cp_offsets, dtype=np.float32)
    cp_order = opt.cp_order
    
    with torch.no_grad():
        for _, data in enumerate(dataloader):
            # get data
            data["source_shape"]    = data["source_shape"].detach().to(opt.device) 
            data["source_cp"]       = data["source_cp"].detach().to(opt.device) 
            source_shape = torch.cat((data["source_shape"], data["source_cp"]), dim=1)
            faces_exists = 'source_face' in data and data['source_face'] is not None

            s_fn = data['source_file'][0]
            s_shape = data["source_shape"][0].clone().cpu()
            s_color = np.array([0.8, 0, 0])
            m_color = np.array([0.5, 0.7, 0.2])
            cp_color = np.array([0.8, 0, 0.8])
            tp_color = np.array([0.8, 0.3, 0.8])

            if faces_exists:
                save_pts(os.path.join(vis_output_dir,"{}-Sa.obj".format(s_fn)), s_shape, colors=s_color, faces=data['source_face'][0].cpu())
            else:
                save_pts(os.path.join(vis_output_dir,"{}-Sa.obj".format(s_fn)), s_shape, colors=s_color)

            save_pts(os.path.join(vis_output_dir,"source_points.obj"), data["source_cp"][0].clone().cpu(), colors=cp_color)

            target_cp = data["source_cp"].clone()[0]
            for idx in range(len(cp_order)):
                target_cp[cp_order[idx]] = target_cp[cp_order[idx]] + cp_offsets[cp_order[idx]]
                
                deformed_classic, _, _ = deform_with_MLS(data["source_cp"], target_cp.unsqueeze(0), source_shape, None, None, alpha=opt.MLS_alpha, deform_type=opt.MLS_deform, epsilon=opt.MLS_eps)

                if faces_exists:
                    save_pts(os.path.join(vis_output_dir,"{}-Sab_CLASSIC_{}.obj".format(s_fn, idx)), deformed_classic[0].cpu(), colors=m_color, faces=data['source_face'][0].cpu())
                else:
                    save_pts(os.path.join(vis_output_dir,"{}-Sab_CLASSIC_{}.obj".format(s_fn, idx)), deformed_classic[0].cpu(), colors=m_color)

                save_pts(os.path.join(vis_output_dir,"target_points_{}.obj".format(idx)), target_cp.clone().cpu(), colors=tp_color)

            # create visualization
            create_temp_visualization(opt, vis_output_dir + os.path.sep, shape=s_fn, cam_radius=4.5, delete_images=False, is_mls_ablation=True)
