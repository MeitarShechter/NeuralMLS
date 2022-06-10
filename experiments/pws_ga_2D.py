import os

import torch
import torch.nn as nn

from modules import WeightNet
from utils.utils import save_pts

def create_2D_illustration_of_piecewise_smoothness_and_geometry_awareness(opt):
    ### declare dataset ###
    opt.num_cp =4 
    net = WeightNet(opt=opt).to(opt.device) # expects (batch_size, 3)
    opt.num_cp = 5
    net2 = WeightNet(opt=opt).to(opt.device) # expects (batch_size, 3)


    ### declare on all relevant losses ###
    hard_const = nn.CrossEntropyLoss()

    ### optimizer ###
    optimizer = torch.optim.Adam([
        {"params": net.parameters()},
        ], lr=opt.lr)
    optimizer2 = torch.optim.Adam([
        {"params": net2.parameters()},
        ], lr=opt.lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, int(opt.nepochs*0.8), gamma=0.1, last_epoch=-1)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, int(opt.nepochs*0.8), gamma=0.1, last_epoch=-1)

    ### train ###
    net.train()
    net2.train()

    os.makedirs(opt.log_dir, exist_ok=True)

    fir = -0.3
    las = 0.3
    grid = torch.linspace(fir,las,30)
    x, y = torch.meshgrid(grid, grid)
    data = torch.cat((x.unsqueeze(-1), y.unsqueeze(-1)), dim=-1).view(x.shape[0]*x.shape[0], 2)
    z = torch.zeros((x.shape[0]*x.shape[0],1))
    shape = torch.cat((data,z), dim=-1).unsqueeze(0)
    cp = torch.tensor([[[fir, fir, 0], [las, fir, 0], [fir, las, 0], [las, las, 0]]])
    cp2 = torch.tensor([[[fir, fir, 0], [las, fir, 0], [fir, las, 0], [0.1, 0.15, 0]]])

    for epoch in range(0, opt.nepochs):
        ############# run network ###########
        optimizer.zero_grad()
        weights = net(cp)
        weights2 = net2(cp2)

        num_cp = cp.shape[1]
        target = torch.arange(num_cp).to(opt.device)
        hard_loss = hard_const(weights[0,:,:], target)

        num_cp2 = cp2.shape[1]
        target2 = torch.arange(num_cp2).to(opt.device)
        hard_loss2 = hard_const(weights2[0,:,:], target2)

        ############# run MLS ##############
        weights = weights / opt.T
        weights = torch.softmax(weights, dim=-1)

        weights2 = weights2 / opt.T
        weights2 = torch.softmax(weights2, dim=-1)

        ############# get losses ###########            
        loss = hard_loss
        loss2 = hard_loss2
        log_str = "Epoch: {:03d}. hard_loss={:.3g}".format(epoch+1, hard_loss.item())
        print(log_str)
        log_str = "Epoch: {:03d}. hard_loss2={:.3g}".format(epoch+1, hard_loss2.item())
        print(log_str)

        loss.backward()
        optimizer.step()
        scheduler.step()

        loss2.backward()
        optimizer2.step()
        scheduler2.step()

    # save weights
    with torch.no_grad():
        weights = net(shape)
        weights = weights / opt.T
        weights = torch.softmax(weights, dim=-1)

        weights2 = net2(shape)
        weights2 = weights2 / opt.T
        weights2 = torch.softmax(weights2, dim=-1)

        norm_weights = weights / torch.max(weights, dim=-2, keepdim=True)[0]
        norm_weights2 = weights2 / torch.max(weights2, dim=-2, keepdim=True)[0]
        for cp_idx in range(cp.shape[1]):
            weight_ours_color = (norm_weights[0,:,cp_idx].unsqueeze(-1).cpu() * torch.tensor([[1, 1, 0]]) + (1-norm_weights[0,:,cp_idx].unsqueeze(-1).cpu()) * torch.tensor([[1, 0, 1]]))
            save_pts(os.path.join(opt.log_dir,"Weights_{}.obj".format(cp_idx)), shape[0], colors=weight_ours_color)
        for cp_idx in range(cp2.shape[1]):
            weight_ours_color = (norm_weights2[0,:,cp_idx].unsqueeze(-1).cpu() * torch.tensor([[1, 1, 0]]) + (1-norm_weights2[0,:,cp_idx].unsqueeze(-1).cpu()) * torch.tensor([[1, 0, 1]]))
            save_pts(os.path.join(opt.log_dir,"Weights_NEW_{}.obj".format(cp_idx)), shape[0], colors=weight_ours_color)


