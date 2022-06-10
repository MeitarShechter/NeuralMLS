import os
import numpy as np
import torch
import torch.nn as nn

from modules import WeightNet
from utils.utils import load_network, save_network, deform_with_MLS, build_dataloader
from utils.utils import deform_with_ARAP, deform_with_KeypointDeformer

from keypoint_deformer.keypointdeformer.models.cage_skinning import CageSkinning
from keypoint_deformer.keypointdeformer.utils.nn import load_network as KPDload_network


def create_user_study(opt):
    # declare dataset
    dataloader = build_dataloader(opt)

    # model declaration
    net = WeightNet(opt=opt).to(opt.device) # expects (batch_size, 3)

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
    hard_const = nn.CrossEntropyLoss()

    ### optimizer ###
    optimizer = torch.optim.Adam([
        {"params": net.parameters()},
        ], lr=opt.lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, int(opt.nepochs*0.8), gamma=0.1, last_epoch=-1)

    ### train ###
    net.train()
    start_epoch = 0 if epoch is None else epoch
    t = 0 if epoch is None else start_epoch*len(dataloader) 

    # log files
    os.makedirs(opt.log_dir, exist_ok=True)

    for epoch in range(start_epoch, opt.nepochs):
        for _, data in enumerate(dataloader):

            ############# get data ###########
            data["source_shape"]    = data["source_shape"].detach().to(opt.device) 

            ### pass is keypoint deformer to get CP ###
            with torch.no_grad():
                KPDoutputs = KPDnet(data["source_shape"].transpose(1,2), target_shape=data["source_shape"].transpose(1,2)) # we don't need the target, but KPD won't work without
            data["source_cp"] = KPDoutputs['source_keypoints'].transpose(1,2)
            if t == 0:
                print("The predicted CP from KPD:\n\n{}\n\n".format(data["source_cp"]))

            ############# run network ###########
            optimizer.zero_grad()

            # create the relevant encoding
            weights = net(data["source_cp"] )

            num_cp = data["source_cp"].shape[1]
            target = torch.arange(num_cp).to(opt.device)
            hard_loss = hard_const(weights[0], target)

            ############# get losses ###########            
            loss = hard_loss
            log_str = "Epoch: {:03d}. t: {:05d}: hard_loss={:.3g}".format(epoch+1, t+1, hard_loss.item())

            print(log_str)

            loss.backward()
            optimizer.step()

            t += 1

        scheduler.step()

    save_network(net, opt.log_dir, network_label="net", epoch_label="final", B_mat=net.cpu().B_mat)

    # create visualization
    data = next(iter(dataloader))
    data["source_shape"] = data["source_shape"].detach().to(opt.device) 
    weights = net(data["source_shape"])
    weights = weights / opt.T
    weights = torch.softmax(weights, dim=-1)

    ### pass is keypoint deformer to get CP ###
    with torch.no_grad():
        KPDoutputs = KPDnet(data["source_shape"].transpose(1,2), target_shape=data["source_shape"].transpose(1,2))
    data["source_cp"] = KPDoutputs['source_keypoints'].transpose(1,2)

    output_dir = opt.log_dir + '/screenshots/'
    os.makedirs(output_dir, exist_ok=True)
    # create visualization
    init_cam_trg = np.array([0., 0., 0.])
    base_fn = output_dir + "{}_{}.png"

    import polyscope as ps
    ps.init()
    ps.set_ground_plane_mode("shadow_only")

    for i in range(3):
        curr_offsets = data["offsets"][:,i,:,:]
        data["target_cp"] = data["source_cp"] + curr_offsets

        deformed_shape, _, _ = deform_with_MLS(data["source_cp"], data["target_cp"], data["source_shape"], None, weights, deform_type=opt.MLS_deform)                
        deformed_classic, _, _ = deform_with_MLS(data["source_cp"], data["target_cp"], data["source_shape"], None, None, alpha=opt.MLS_alpha, deform_type=opt.MLS_deform)
        deformed_KPD, _ = deform_with_KeypointDeformer(data["source_cp"], data["target_cp"], data["source_shape"], KPDoutputs)

        s_shape = data["source_shape"][0].cpu()
        source = s_shape.numpy()
        s_max = np.max(source, axis=0)
        s_min = np.min(source, axis=0)
        s_mid = (s_max + s_min)/2
        color = np.tile(np.array([[0.7, 0.2, 0.5]]).astype('float32'), (source.shape[0],1))
        neuralMLS = deformed_shape[0].detach().cpu().double().numpy()
        neuralMLS = neuralMLS - s_mid
        MLS = deformed_classic[0].detach().cpu().double().numpy()    
        MLS = MLS - s_mid            
        KPD = deformed_KPD[0].cpu().double().numpy()
        KPD = KPD - s_mid
        cp = data["source_cp"][0].cpu().double().numpy()
        cp = cp - s_mid
        tp = data["target_cp"][0].cpu().double().numpy()
        tp = tp - s_mid

        ## register meshes
        # ours
        face = data['source_face'][0].cpu().numpy()
        # Position the camera
        init_cam_pos = np.array(opt.camera_pos[0])
        ps.look_at(init_cam_pos, init_cam_trg)

        # cp_c = ps.register_point_cloud("cp", cp, radius=10, enabled=True)
        # tp_c = ps.register_point_cloud("tp", tp, radius=5, enabled=True)

        file_name = base_fn.format("neuralMLS", i)
        neuralMLS_c = ps.register_surface_mesh("neuralMLS", neuralMLS, face, enabled=True)
        neuralMLS_c.add_color_quantity("neuralMLS colors", color, enabled=True)        
        ps.screenshot(filename=file_name, transparent_bg=False)
        neuralMLS_c.set_enabled(False)

        file_name = base_fn.format("MLS", i)
        MLS_c = ps.register_surface_mesh("MLS", MLS, face, enabled=True)
        MLS_c.add_color_quantity("MLS colors", color, enabled=True)
        ps.screenshot(filename=file_name, transparent_bg=False)
        MLS_c.set_enabled(False)

        file_name = base_fn.format("KPD", i)        
        KPD_c = ps.register_surface_mesh("KPD", KPD, face, enabled=True)
        KPD_c.add_color_quantity("KPD colors", color, enabled=True)
        ps.screenshot(filename=file_name, transparent_bg=False)
        KPD_c.set_enabled(False)
