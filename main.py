import datetime
import warnings
import os
import torch
import torch.nn as nn
import numpy as np

from modules import WeightNet
from utils.utils import BaseOptions, load_network, save_network, deform_with_MLS, build_dataloader
from utils.utils import save_pts, log_outputs 


def train(opt):
    ### declare dataset ###
    dataloader = build_dataloader(opt)

    ### model declaration ###
    net = WeightNet(opt=opt).to(opt.device) # expects (batch_size, in_dim)

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
            data["source_cp"] = data["source_cp"].detach().to(opt.device) 
            num_cp = data["source_cp"].shape[1]

            ############# run network ###########
            optimizer.zero_grad()
            weights = net(data["source_cp"])
            target = torch.arange(num_cp).to(opt.device)
            loss = ce_loss(weights[0], target)

            ############# print and log ###########            
            log_str = "Epoch: {:03d}. t: {:05d}: ce_loss={:.3g}".format(epoch+1, t+1, loss.item())
            print(log_str)
            log_file.write(log_str+"\n")
            if (t + 1) % save_interval == 0:
                log_outputs(opt, t+1, net, data, save_all=(t+1==save_interval))

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

def deform_shape(opt, net=None, save_subdir="test"):
    opt.batch_size = 1
    dataloader = build_dataloader(opt)

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
            ############# get data ###########
            data["source_shape"]    = data["source_shape"].detach().to(opt.device) 
            data["source_cp"]       = data["source_cp"].detach().to(opt.device) 
            data["target_cp"]       = data["target_cp"].detach().to(opt.device) 

            # add CP to the shapes for our case
            source_shape = data["source_shape"]

            ############# run network ###########
            weights = net(source_shape)

            ############# run MLS ##############
            weights = weights / opt.T
            weights = torch.softmax(weights, dim=-1)

            deformed_shape, _, _ = deform_with_MLS(data["source_cp"], data["target_cp"], source_shape, None, weights, deform_type=opt.MLS_deform)

            # Source
            s_fn = data['source_file'][0]
            s_shape = data["source_shape"][0].cpu()
            if 'source_face' in data and data['source_face'] is not None:
                save_pts(os.path.join(test_output_dir,"{}-Sa.obj".format(s_fn)), s_shape, colors=np.array([0.8, 0, 0]), faces=data['source_face'][0].cpu())
            else:
                save_pts(os.path.join(test_output_dir,"{}-Sa.obj".format(s_fn)), s_shape, colors=np.array([0.8, 0, 0]))
            # Deformed
            if 'source_face' in data and data['source_face'] is not None:
                save_pts(os.path.join(test_output_dir,"{}-Sab.obj".format(s_fn)), deformed_shape[0].cpu(), colors=np.array([0, 0, 0.8]), faces=data['source_face'][0].cpu())
            else:
                save_pts(os.path.join(test_output_dir,"{}-Sab.obj".format(s_fn)), deformed_shape[0].cpu(), colors=np.array([0, 0, 0.8]))
            # Control points
            save_pts(os.path.join(test_output_dir,"source_points.obj"), data["source_cp"][0].cpu(), colors=np.array([0.8, 0, 0.8]))
            save_pts(os.path.join(test_output_dir,"target_points.obj"), data["target_cp"][0].cpu(), colors=np.array([0.8, 0.3, 0.8]))


if __name__ == "__main__":
    parser = BaseOptions()
    opt = parser.parse()

    if opt.ckpt is not None:
        opt.log_dir = os.path.dirname(opt.ckpt)
    else:
        opt.log_dir = os.path.join(opt.log_dir, "-".join(filter(None, [os.path.basename(__file__)[:-3],
                                                         datetime.datetime.now().strftime("%d-%m-%Y__%H:%M:%S"),
                                                        opt.name]))) # log dir will be the file name + date_time + expirement name

    ### setup device ###
    if opt.not_cuda:
        device = torch.device("cpu")
    else:
        if not torch.cuda.is_available():
            warnings.warn("WARNING: Cuda is not available, using cpu instead")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    opt.device = device

    if opt.phase == "deform_shape":
        deform_shape(opt, save_subdir=opt.subdir)

    else:
        os.makedirs(opt.log_dir, exist_ok=True)
        log_file = open(os.path.join(opt.log_dir, "training_log.txt"), "a")
        parser.print_options(opt, log_file)
        log_file.close()
        train(opt)


