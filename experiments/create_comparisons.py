import os
import numpy as np

import pymesh
import torch
import torch.nn as nn

from modules import WeightNet
from utils.utils import load_network, save_network, deform_with_MLS, build_dataloader, group_knn
from utils.utils import save_pts, log_outputs, deform_with_KeypointDeformer, deform_with_ARAP
from visualizations.create_visualization import create_visualization 
from keypoint_deformer.keypointdeformer.models.cage_skinning import CageSkinning
from keypoint_deformer.keypointdeformer.utils.nn import load_network as KPDload_network


def create_comparisons(opt):
    ### declare dataset ###
    dataloader = build_dataloader(opt)

    ### model declaration ###
    net = WeightNet(opt=opt).to(opt.device) # expects (batch_size, in_dim)

    ### model declaration for keypointDeformer###
    KPDnet = CageSkinning(opt)
    KPDckpt = opt.KPDckpt
    KPDload_network(KPDnet, KPDckpt)
    KPDnet.eval()

    ### load checkpoint if in need ###
    epoch = None
    if opt.ckpt:
        net, epoch = load_network(net, opt.ckpt, opt.device)

    ### declare on all relevant losses ###
    ce_loss = nn.CrossEntropyLoss()

    ### optimizer ###
    optimizer = torch.optim.Adam([
        {"params": net.parameters()},
        ], lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, int(opt.nepochs*0.8), gamma=0.1, last_epoch=-1)

    ### misc ###
    log_file = open(os.path.join(opt.log_dir, "training_log.txt"), "a")
    log_file.write("----- Starting training process -----\n")
    log_file.write(str(net)+"\n")
    net.train()
    start_epoch = 0 if epoch is None else epoch
    assert(start_epoch < opt.nepochs), "Cannot have start epoch bigger than total number of epoches to train on"
    t = 0 if epoch is None else start_epoch*len(dataloader) 
    save_interval = max(opt.nepochs//3, 1)

    for epoch in range(start_epoch, opt.nepochs):
        for _, data in enumerate(dataloader):

            ############# get data ###########
            data["source_shape"]    = data["source_shape"].detach().to(opt.device) 

            ### pass is keypoint deformer to get CP ###
            with torch.no_grad():
                KPDoutputs = KPDnet(data["source_shape"].transpose(1,2), target_shape=data["source_shape"].transpose(1,2))
            data["source_cp"] = KPDoutputs['source_keypoints'].transpose(1,2)
            if t == 0:
                print("The predicted CP from KPD:\n\n{}\n\n".format(data["source_cp"]))
            num_cp = data["source_cp"].shape[1]

            ############# run network ###########
            optimizer.zero_grad()

            weights = net(data["source_cp"])
            target = torch.arange(num_cp).to(opt.device)
            loss = ce_loss(weights[0], target)

            ############# get losses ###########            
            log_str = "Epoch: {:03d}. t: {:05d}: hard_loss={:.3g}".format(epoch+1, t+1, loss.item())

            print(log_str)
            log_file.write(log_str+"\n")
            if (t + 1) % save_interval == 0:
                log_outputs(opt, t+1, net, data, save_all=(t+1==save_interval), KPDoutputs=KPDoutputs)

            loss.backward()
            optimizer.step()
                         
            if (t + 1) % save_interval == 0:
                save_network(net, opt.log_dir, network_label="net", epoch_label="latest", device=opt.device, epoch=epoch, B_mat=net.cpu().B_mat)

            t += 1

        if (epoch + 1) % save_interval == 0:
            save_network(net, opt.log_dir, network_label="net", epoch_label=epoch+1, device=opt.device, epoch=epoch, B_mat=net.cpu().B_mat)

        scheduler.step()

    log_file.close()
    save_network(net, opt.log_dir, network_label="net", epoch_label="final", B_mat=net.cpu().B_mat)

    print("Done training, now starting to create the comparisons")

    create_visualization_of_comparisons(opt, net=net, KPDnet=KPDnet)

def create_visualization_of_comparisons(opt, net=None, KPDnet=None, save_subdir="NeuralMLS_comparisons"):
    opt.batch_size = 1
    dataloader = build_dataloader(opt)
    assert (KPDnet is not None), "Can't create visualization of comparisons without KPD network"

    if net is None:
        # network
        net = WeightNet(opt=opt).to(opt.device) # expects (batch_size, 3)
        net.eval()
        load_network(net, opt.ckpt, opt.device)
    else:
        net.eval()

    test_output_dir = os.path.join(opt.log_dir, save_subdir)
    os.makedirs(test_output_dir, exist_ok=True)
    
    with torch.no_grad():
        for _, data in enumerate(dataloader):
            assert ('source_face' in data and data['source_face'] is not None), "Can't run ARAP and can't calculate metrics without faces. Choose shapes with faces for the comparisons"

            ############# get data ###########
            data["source_shape"] = data["source_shape"].detach().to(opt.device) 

            ### pass in keypoint deformer to get CP ###
            with torch.no_grad():
                KPDoutputs = KPDnet(data["source_shape"].transpose(1,2), target_shape=data["source_shape"].transpose(1,2))
            data["source_cp"] = KPDoutputs['source_keypoints'].transpose(1,2)
            data["target_cp"] = data["source_cp"] + data["offsets"]

            source_shape = data["source_shape"]
            weights = net(source_shape)
            weights = weights / opt.T
            weights = torch.softmax(weights, dim=-1)

            deformed_NeuralMLS, _, _    = deform_with_MLS(data["source_cp"], data["target_cp"], source_shape, None, weights, deform_type=opt.MLS_deform)
            deformed_MLS, _, _          = deform_with_MLS(data["source_cp"], data["target_cp"], source_shape, None, None, deform_type=opt.MLS_deform, alpha=opt.MLS_alpha)
            deformed_KPD, _             = deform_with_KeypointDeformer(data["source_cp"], data["target_cp"], data["source_shape"], KPDoutputs)
            run_arap = False
            try:
                deformed_ARAP = deform_with_ARAP(data["source_cp"], data["target_cp"], source_shape, data['source_face'])
                run_arap = True
            except RuntimeError as e:                
                print("Failed to run ARAP:\n{}".format(e))
                run_arap = False

            # Log
            s_fn = data['source_file'][0]
            s_shape = data["source_shape"][0].cpu()
            save_pts(os.path.join(test_output_dir,"{}-Sa.obj".format(s_fn)), s_shape, colors=np.array([0.84, 0.37, 0]), faces=data['source_face'][0].cpu())
            save_pts(os.path.join(test_output_dir,"{}-Sab.obj".format(s_fn)), deformed_NeuralMLS[0].cpu(), colors=np.array([0, 0.45, 0.7]), faces=data['source_face'][0].cpu())
            save_pts(os.path.join(test_output_dir,"{}-Sab_MLS.obj".format(s_fn)), deformed_MLS[0].cpu(), colors=np.array([0.5, 0.7, 0.2]), faces=data['source_face'][0].cpu())
            save_pts(os.path.join(test_output_dir,"{}-Sab_KPD.obj".format(s_fn)), deformed_KPD[0].cpu(), colors=np.array([0.4, 0, 0.8]), faces=data['source_face'][0].cpu())
            if run_arap:
                save_pts(os.path.join(test_output_dir,"{}-Sab_ARAP.obj".format(s_fn)), deformed_ARAP, colors=np.array([0.7, 0.2, 0.5]), faces=data['source_face'][0].cpu())
            save_pts(os.path.join(test_output_dir,"source_points.obj"), data["source_cp"][0].cpu(), colors=np.array([0.8, 0, 0.8]))
            save_pts(os.path.join(test_output_dir,"target_points.obj"), data["target_cp"][0].cpu(), colors=np.array([0.2, 0.2, 0.2]))

            ############# get metrics ###########  
            s_face = data["source_face"][0].numpy()

            source_pm    = pymesh.form_mesh(s_shape.detach().numpy(), s_face)
            neuralMLS_pm = pymesh.form_mesh(deformed_NeuralMLS[0].detach().numpy(), s_face)
            MLS_pm       = pymesh.form_mesh(deformed_MLS[0].detach().numpy(), s_face)
            KPD_pm       = pymesh.form_mesh(deformed_KPD[0].detach().numpy(), s_face)
            if run_arap:
                ARAP_pm  = pymesh.form_mesh(deformed_ARAP.detach().numpy(), s_face)

            log_str = "---- Laplacian results: -----\n"

            source_pm.add_attribute("vertex_laplacian")
            neuralMLS_pm.add_attribute("vertex_laplacian")
            MLS_pm.add_attribute("vertex_laplacian")
            KPD_pm.add_attribute("vertex_laplacian")
            if run_arap:
                ARAP_pm.add_attribute("vertex_laplacian")
            
            s_lap           = source_pm.get_vertex_attribute("vertex_laplacian")
            s_lap_norm      = np.linalg.norm(s_lap, axis=-1)
            
            neuralMLS_lap       = neuralMLS_pm.get_vertex_attribute("vertex_laplacian")
            neuralMLS_lap_diff  = np.abs(s_lap_norm - np.linalg.norm(neuralMLS_lap, axis=-1))
            log_str = "{}NeuralMLS: {}\n".format(log_str, np.nanmean(neuralMLS_lap_diff))
            
            MLS_lap             = MLS_pm.get_vertex_attribute("vertex_laplacian")
            MLS_lap_diff        = np.abs(s_lap_norm - np.linalg.norm(MLS_lap, axis=-1))
            log_str = "{}MLS: {}\n".format(log_str, np.nanmean(MLS_lap_diff))

            KPD_lap             = KPD_pm.get_vertex_attribute("vertex_laplacian")
            KPD_lap_diff        = np.abs(s_lap_norm - np.linalg.norm(KPD_lap, axis=-1))
            log_str = "{}KPD: {}\n".format(log_str, np.nanmean(KPD_lap_diff))

            if run_arap:
                ARAP_lap = ARAP_pm.get_vertex_attribute("vertex_laplacian")
                ARAP_lap_diff = np.abs(s_lap_norm - np.linalg.norm(ARAP_lap, axis=-1))
                log_str = "{}ARAP: {}\n".format(log_str, np.nanmean(ARAP_lap_diff))

            print(log_str)
            log_file = open(os.path.join(opt.log_dir, "log_test.txt"), "a")
            log_file.write("{}\n".format(log_str))
                    
            log_str = "---- Mean curevature results: -----\n"

            source_pm.add_attribute("vertex_mean_curvature")
            neuralMLS_pm.add_attribute("vertex_mean_curvature")
            MLS_pm.add_attribute("vertex_mean_curvature")
            KPD_pm.add_attribute("vertex_mean_curvature")
            if run_arap:
                ARAP_pm.add_attribute("vertex_mean_curvature")

            s_curv = np.abs(source_pm.get_vertex_attribute("vertex_mean_curvature"))

            neuralMLS_curv = neuralMLS_pm.get_vertex_attribute("vertex_mean_curvature")
            neuralMLS_curv_diff = np.abs(s_curv - np.abs(neuralMLS_curv))
            log_str = "{}neuralMLS: {}\n".format(log_str, np.nanmean(neuralMLS_curv_diff))

            MLS_curv = MLS_pm.get_vertex_attribute("vertex_mean_curvature")
            MLS_curv_diff = np.abs(s_curv - np.abs(MLS_curv))
            log_str = "{}MLS: {}\n".format(log_str, np.nanmean(MLS_curv_diff))

            KPD_curv = KPD_pm.get_vertex_attribute("vertex_mean_curvature")
            KPD_curv_diff = np.abs(s_curv - np.abs(KPD_curv))
            log_str = "{}KPD: {}\n".format(log_str, np.nanmean(KPD_curv_diff))

            if run_arap:
                ARAP_curv = ARAP_pm.get_vertex_attribute("vertex_mean_curvature")
                ARAP_curv_diff = np.abs(s_curv - np.abs(ARAP_curv))
                log_str = "{}ARAP: {}\n".format(log_str, np.nanmean(ARAP_curv_diff))
            
            print(log_str)
            log_file.write("{}\n".format(log_str))

            log_str = "---- L2 results: -----\n"

            _, knn_idx, _ = group_knn(1, data["source_cp"], source_shape, NCHW=False) # (B, P, K, D), (B, P, K)
            
            neuralMLS_points = deformed_NeuralMLS[0, knn_idx[0, :, 0], :] #(P, D)
            neuralMLS_dist = torch.linalg.norm(neuralMLS_points - data["target_cp"][0], ord=2, dim=-1)
            log_str = "{}neuralMLS: {}\n".format(log_str, torch.mean(neuralMLS_dist))

            MLS_points = deformed_MLS[0, knn_idx[0, :, 0], :] #(P, D)
            MLS_dist = torch.linalg.norm(MLS_points - data["target_cp"][0], ord=2, dim=-1)
            log_str = "{}MLS: {}\n".format(log_str, torch.mean(MLS_dist))

            KPD_points = deformed_KPD[0, knn_idx[0, :, 0], :] #(P, D)
            KPD_dist = torch.linalg.norm(KPD_points - data["target_cp"][0], ord=2, dim=-1)
            log_str = "{}KPD: {}\n".format(log_str, torch.mean(KPD_dist))

            if run_arap:
                ARAP_points = deformed_ARAP[knn_idx[0, :, 0], :] #(P, D)
                ARAP_dist = np.linalg.norm(ARAP_points - data["target_cp"][0].numpy(), axis=-1)
                log_str = "{}ARAP: {}\n".format(log_str, np.mean(ARAP_dist))
            
            print(log_str)
            log_file.write("{}\n".format(log_str))

            log_file.close()
            
            # create visualization
            create_visualization(test_output_dir + os.path.sep, shape=s_fn, cam_radius=3.5, num_images=32, is_test=True, with_faces=True)