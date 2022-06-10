import os

import torch
import torch.nn as nn

from modules import WeightNet
from utils.utils import save_1d_pts, deform_with_MLS_1D

def create_1D_illustration_of_piecewise_smoothness_and_geometry_awareness(opt):
    ### declare dataset ###
    opt.num_cp = 6
    net = WeightNet(opt=opt, in_dim=1).to(opt.device) # expects (batch_size, in_dim)

    ### declare on all relevant losses ###
    hard_const = nn.CrossEntropyLoss()

    ### optimizer ###
    optimizer = torch.optim.Adam([
        {"params": net.parameters()},
        ], lr=opt.lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, int(opt.nepochs*0.8), gamma=0.1, last_epoch=-1)

    ### train ###
    net.train()

    os.makedirs(opt.log_dir, exist_ok=True)

    fir = -5
    las = 5
    x = torch.linspace(fir, las, 800+1)
    shape = x.unsqueeze(-1)
    num_cp = 6
    cp = torch.linspace(fir*0.8, las*0.8, num_cp).unsqueeze(-1)

    cp = cp + torch.zeros_like(cp).normal_(0, 0.7)
    target_cp = torch.cat([torch.linspace(fir, (fir+las)/2, num_cp//2), torch.linspace((fir+las)/2, las, num_cp//2)], dim=0).unsqueeze(-1)

    if opt.do_geometric_awareness:
        net2 = WeightNet(opt=opt, in_dim=1).to(opt.device) # expects (batch_size, in_dim)
        optimizer2 = torch.optim.Adam([
        {"params": net2.parameters()},
        ], lr=opt.lr)
        scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, int(opt.nepochs*0.8), gamma=0.1, last_epoch=-1)
        net2.train()

        n_cp = torch.zeros_like(cp)
        n_cp[num_cp//2] = -1
        n_cp = cp + n_cp

        for epoch in range(0, opt.nepochs):
            ############# run network ###########
            optimizer2.zero_grad()
            weights = net2(n_cp)

            target = torch.arange(num_cp).to(opt.device)
            hard_loss = hard_const(weights, target)

            ############# run MLS ##############
            weights = weights / opt.T
            weights = torch.softmax(weights, dim=-1)

            ############# get losses ###########            
            loss = hard_loss
            log_str = "Epoch: {:03d}. hard_loss={:.3g}".format(epoch+1, hard_loss.item())
            print(log_str)

            loss.backward()
            optimizer2.step()
            scheduler2.step()

    for epoch in range(0, opt.nepochs):
        ############# run network ###########
        optimizer.zero_grad()
        weights = net(cp)

        target = torch.arange(num_cp).to(opt.device)
        hard_loss = hard_const(weights, target)

        ############# run MLS ##############
        weights = weights / opt.T
        weights = torch.softmax(weights, dim=-1)

        ############# get losses ###########            
        loss = hard_loss
        log_str = "Epoch: {:03d}. hard_loss={:.3g}".format(epoch+1, hard_loss.item())
        print(log_str)

        loss.backward()
        optimizer.step()
        scheduler.step()

    # save weights
    with torch.no_grad():
        weights = net(shape)
        weights = weights / opt.T
        weights = torch.softmax(weights, dim=-1)

        if opt.do_geometric_awareness:
            weights2 = net2(shape)
            weights2 = weights2 / opt.T
            weights2 = torch.softmax(weights2, dim=-1)

            norm_weights2 = weights2

        deformed_shape, _, dist_w = deform_with_MLS_1D(cp, cp + target_cp, shape, weights, alpha=opt.MLS_alpha, eps=opt.MLS_eps)
        MLS_deformed_shape, _, _ = deform_with_MLS_1D(cp, cp + target_cp, shape, None, alpha=opt.MLS_alpha, eps=opt.MLS_eps)

        norm_weights = weights
        colors = torch.tensor([
            [0, 0, 0],
            [1, 0, 0],
            [0.5, 0, 0],
            [0.6, 0.3, 0],
            [0.5, 0.6, 0],
            [0, 0.6, 0.6] 
            ])
        # do weights visualization
        if opt.do_geometric_awareness:
            _, _, dist_w2 = deform_with_MLS_1D(n_cp, n_cp + target_cp, shape, None, alpha=opt.MLS_alpha, eps=opt.MLS_eps)
            save_1d_pts(os.path.join(opt.log_dir,"NeuralMLS&MLS_before&After_weights_visualization.png"), shape, offsets=norm_weights, offsets2=norm_weights2, colors=colors, cp=cp, cp2=n_cp, o3=dist_w, o4=dist_w2)
        else:        
            save_1d_pts(os.path.join(opt.log_dir,"NeuralMLS_weights_visualization.png"), shape, offsets=norm_weights, colors=colors, cp=cp)
            # do the same for MLS weights
            save_1d_pts(os.path.join(opt.log_dir,"MLS_weights_visualization.png"), shape, offsets=dist_w, colors=colors, cp=cp)

        # do offsets visualization
        colors = torch.tensor([0, 0, 0])
        save_1d_pts(os.path.join(opt.log_dir,"NeuralMLS_offsets_visualization.png"), shape, offsets=deformed_shape - shape, colors=colors, cp=cp, t_cp=target_cp)
        # do the same for MLS offsets
        save_1d_pts(os.path.join(opt.log_dir,"MLS_offsets_visualization.png"), shape, offsets=MLS_deformed_shape - shape, colors=colors, cp=cp, t_cp=target_cp)
