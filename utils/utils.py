import argparse
import os
import io
import torch
import numpy as np
import matplotlib.colors as mpc
from collections import OrderedDict

from datasets import SingleShapeDataset
from keypoint_deformer.keypointdeformer.utils.cages import deform_with_MVC

import scipy.sparse.linalg as spla
from matplotlib import pyplot as plt 
from utils.arap import ARAP
import robust_laplacian


class BaseOptions():
    """
    This class defines options used during both training and test time.
    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument("--name", help="experiment name")
        parser.add_argument("--dataset", type=str, help="dataset name", default="SINGLE_SHAPE_TEST", choices=["SINGLE_SHAPE_TEST"])
        parser.add_argument("--num_data_examples", type=int, help="number of data example to use", default=-1)
        parser.add_argument("--num_points", type=int, help="number of input points", default=2048)
        parser.add_argument("--num_cp_to_use", type=int, help="temptemptemp", default=10)
        parser.add_argument("--source_model", type=str, nargs="*", help="source model for testing")
        parser.add_argument("--target_model", type=str, nargs="*", help="target model used for testing")
        parser.add_argument("--data_dir", type=str, help="data root", default="~/Git/Dynamic-Neural-Cages/data/")
        parser.add_argument("--dim", type=int, help="2D or 3D", default=3)
        parser.add_argument("--log_dir", type=str, help="log directory", default="./log")
        parser.add_argument("--subdir", type=str, help="save to directory name", default="results")
        parser.add_argument("--batch_size", type=int, help="batch size", default=1)
        parser.add_argument("--not_cuda", action="store_true", help="whether to use cuda device or cpu")
        parser.add_argument("--en_pos_enc", action="store_true", help="whether to use positional encoding")
        parser.add_argument("--MLS_alpha", type=float, help="alpha reg for MLS", default=0.5)        
        parser.add_argument("--MLS_eps", type=float, help="epsilon reg for MLS", default=1e-8)        
        parser.add_argument("--MLS_deform", type=str, help="which deformation to apply", default="affine", choices=["affine", "similarity", "rigid"])        
        parser.add_argument("--deform_shape", action="store_true", help="whether we also do shape deformation after training")
        parser.add_argument("-cpoff", "--cp_offsets", type=float, nargs="*", action='append', help="where to more each CP (for visualization mode)")
        parser.add_argument("-cpord", "--cp_order", type=int, nargs="*", action='append', help="the order in which we move them (for visualization mode)")
        parser.add_argument("-campos", "--camera_pos", type=float, nargs="*", action='append', help="the positions of the camera (for visualization mode)")
        parser.add_argument("--cam_radius", type=float, help="radius for camera", default=4.5)
        parser.add_argument("--temps", type=float, nargs="*", help="the temperature to visualize for temp_vis mode")
        parser.add_argument("--alphas", type=float, nargs="*", help="the alphas to visualize for mls_ablation mode")
        # regularizations
        parser.add_argument("--lr", type=float, help="learning rate", default=3e-4)
        parser.add_argument("--K", type=int, help="number of nearest neightbors to consider when necessary", default=16)
        parser.add_argument("--T", type=float, help="softmax temperatre", default=1)
        # training setup
        parser.add_argument("--nepochs", type=int, help="total number of epochs", default=50)
        parser.add_argument("--phase", type=str, choices=["train", "deform_shape", "visualize", "create_comparisons", "create_temperature_illustration", "create_user_study", "mls_ablation", "create_1d_illustration", "create_2d_illustration"], default="train")
        parser.add_argument("--ckpt", type=str, help="path to model to test")
        parser.add_argument("--KPDckpt", type=str, help="path to weight model to test")
        parser.add_argument("--do_geometric_awareness", action="store_true", help="")
        # network options
        parser.add_argument("--bottleneck_size", type=int, help="bottleneck size", default=256)
        parser.add_argument("--PE_sigma", type=float, help="std of PE gaussian", default=1)

        # Keypoint Deformer
        parser.add_argument("--n_keypoints", type=int, help="")
        parser.add_argument("--ico_sphere_div", type=int, help="", default=1)
        parser.add_argument("--n_influence_ratio", type=float, help="", default=1.0)
        parser.add_argument("--cage_size", type=float, default=1.4, help="")
        parser.add_argument("--n_fps", type=int, help="")
        parser.add_argument("--no_optimize_cage", action="store_true", help="")
        parser.add_argument("--normalization", type=str, help="normalization layer for KPD", default=None)        
        parser.add_argument("--num_point", type=int, help="number of points for KPD", default=1024)
        parser.add_argument("--disable_c_residual", dest="c_residual", action="store_false")
        parser.add_argument("--disable_d_residual", dest="d_residual", action="store_false")
        self.initialized = True
        return parser

    def gather_options(self):
        """
        Initialize our parser with basic options (only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='dynamic cage deformation')
            parser = self.initialize(parser)

            # save and return the parser
            self.parser = parser
            # get the basic options
            opt, _ = self.parser.parse_known_args()

        return self.parser.parse_args()

    def print_options(self, opt, output_file=None):
        """
        Print and save options
        It will print both current options and default values (if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        # print(message)

        # save to the disk
        if isinstance(output_file, str):
            with open(output_file, "a") as f:
                f.write(message)
                f.write('\n')
        elif isinstance(output_file, io.IOBase):
            output_file.write(message)
            output_file.write('\n')

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        if opt.phase == "test":
            assert(opt.ckpt is not None)
        opt.batch_size = opt.batch_size if opt.phase=="train" else 1
        self.opt = opt
        if isinstance(opt.source_model, str):
            opt.source_model = [opt.source_model]

        return self.opt


def load_network(net, path, device):
    """
    load network parameters whose name exists in the pth file.
    return:
        INT trained step
    """
    epoch = None
    B_mat = None

    if isinstance(path, str):
        print("loading network from {}".format(path))
        if path[-3:] == "pth":
            loaded_state = torch.load(path, map_location=device)
            if "epoch" in loaded_state:
                epoch = loaded_state["epoch"] 
            if "states" in loaded_state:
                loaded_state = loaded_state["states"]
            if "B_mat" in loaded_state:
                B_mat = loaded_state["B_mat"] 
        else:
            loaded_state = np.load(path).item()
            if "epoch" in loaded_state:
                epoch = loaded_state["epoch"]
            if "states" in loaded_state:
                loaded_state = loaded_state["states"]
            if "B_mat" in loaded_state:
                B_mat = loaded_state["B_mat"]
    elif isinstance(path, dict):
        loaded_state = path

    network = net.module if isinstance(net, torch.nn.DataParallel) else net

    missingkeys, unexpectedkeys = network.load_state_dict(loaded_state, strict=False)
    if B_mat is not None:
        network.load_B_mat(B_mat)

    if len(missingkeys)>0:
        print("load_network {} missing keys".format(len(missingkeys)), "\n".join(missingkeys))
    if len(unexpectedkeys)>0:
        print("load_network {} unexpected keys".format(len(unexpectedkeys)), "\n".join(unexpectedkeys))

    return network, epoch

def save_network(net, directory, network_label, epoch_label=None, device=None, **kwargs):
    """
    save model to directory with name {network_label}_{epoch_label}.pth
    Args:
        net: pytorch model
        directory: output directory
        network_label: str
        epoch_label: convertible to str
        kwargs: additional value to be included
    """
    save_filename = "_".join((network_label, str(epoch_label))) + ".pth"
    save_path = os.path.join(directory, save_filename)
    merge_states = OrderedDict()
    merge_states["states"] = net.cpu().state_dict()    
    for k in kwargs:
        merge_states[k] = kwargs[k]
    torch.save(merge_states, save_path)
    if device is not None:
        net = net.to(device)
    else:
        if torch.cuda.is_available():
            net = net.cuda()

def save_pts(filename, points, normals=None, labels=None, colors=None, faces=None):
    assert(points.ndim==2)
    if points.shape[-1] == 2:
        points = np.concatenate([points, np.zeros_like(points)[:, :1]], axis=-1)
    if normals is not None:
        points = np.concatenate([points, normals], axis=1)
    if labels is not None:
        f = " %f" * points.shape[1]
        points = np.concatenate([points, labels], axis=1)
        f += " %d"
        # np.savetxt(filename, points, fmt=["%.10e"]*points.shape[1]+["\"%i\""])
        np.savetxt(filename, points, fmt="v{}".format(f))
        return
    if colors is not None:
        points = points.cpu().detach().numpy()
        if colors.ndim == 1:
            colors = np.tile(colors, (points.shape[0], 1))
        points = np.concatenate([points, colors], axis=1)
        f = " %f" * points.shape[1]
        np.savetxt(filename, points, fmt="v{}".format(f))
        if faces is not None:
            with open(filename, "ab") as f:
                f.write(b"\n")
                f2 = " %d" * faces.shape[1]
                np.savetxt(f, faces+1, fmt="f{}".format(f2))
    else:
        # np.savetxt(filename, points, fmt=["%.10e"]*points.shape[1])
        np.savetxt(filename, points.cpu().detach().numpy())

def save_1d_pts(filename, points, colors=None, offsets=None, offsets2=None, cp=None, cp2=None, t_cp=None, o3=None, o4=None):
    points = points.cpu().detach().numpy()
    do_ylim = False
    if colors is None:
        colors = np.array([0, 0, 0])
    if colors.ndim == 1:
        colors = np.tile(colors, (points.shape[0], 1))
    if colors.shape[0] != points.shape[0] and offsets is not None and offsets.shape[1] == colors.shape[0]:
        print("Doing weights visualization run!")
        # we're in weights visualization run
        num_cp = colors.shape[0]
        orig_colors = colors.clone()
        colors = torch.repeat_interleave(colors, points.shape[0], dim=0)
        points = np.tile(points, (num_cp, 1))
        offsets = offsets.T.flatten()
        if offsets2 is not None:
            offsets2 = offsets2.T.flatten() 
            o3 = o3.T.flatten() 
            o4 = o4.T.flatten() 
        s = np.array([10] * points.shape[0])

        # # add cp to the visualization
        # points = np.concatenate((points, cp), axis=0)
        # offsets = np.concatenate((offsets, np.array([-0.1]*cp.shape[0])))
        # colors = torch.cat([colors, orig_colors], dim=0)
        # s = np.concatenate((s, np.array([20]*cp.shape[0])), axis=0)
        do_ylim = True
    else:    
        if offsets is None:
            offsets = np.array([0])
        if offsets.ndim == 1:
            offsets = np.tile(offsets, (points.shape[0], 1))
        else:
            s = np.array([10] * points.shape[0])
            # # add ground truth offsets
            # points = np.concatenate((points, cp), axis=0)
            # offsets = np.concatenate((offsets, t_cp), axis=0)
            # # gt_colors = torch.tensor([1, 0, 0]).tile((t_cp.shape[0], 1))
            # # colors = torch.cat([torch.from_numpy(colors), gt_colors], dim=0)
            colors = offsets
            # s = np.concatenate((s, np.array([40]*cp.shape[0])), axis=0)

    # points = np.concatenate([points, offsets], axis=1)
    normalize = mpc.Normalize(vmin=-6, vmax=6)

    if offsets2 is not None:
        # fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)
        fig, (ax1, ax2) = plt.subplots(2, 2, sharex=True, sharey=False, figsize=(16,5))
        ax1[0].scatter(points, offsets, s=s, c=colors, norm=normalize)
        ax1[0].axhline(y=0, color='black', linestyle='--')
        ax1[0].scatter(cp, np.array([-0.1]*cp.shape[0]), s=np.array([100]*cp.shape[0]), c=orig_colors, marker='*')
        ax1[0].set_ylim(-0.2, 1.1)
        ax2[0].scatter(points, offsets2, s=s, c=colors, norm=normalize)
        ax2[0].axhline(y=0, color='black', linestyle='--')
        ax2[0].scatter(cp2, np.array([-0.1]*cp2.shape[0]), s=np.array([100]*cp2.shape[0]), c=orig_colors, marker='*')
        ax2[0].set_ylim(-0.2, 1.1)
        ax2[0].set_xlabel("point location", fontsize=16)
        ax2[1].set_xlabel("point location", fontsize=16)
        # ax1.set_ylabel("Configuration 1 - weight")
        # ax2.set_ylabel("Configuration 2 - weight")
        ax2[0].set_ylabel("weight", fontsize=24)
        # ax2.set_ylabel("weight", fontsize=24)

        ax1[1].scatter(points, o3, s=s, c=colors, norm=normalize)
        ax1[1].axhline(y=0, color='black', linestyle='--')
        ax1[1].scatter(cp, np.array([-1.5]*cp.shape[0]), s=np.array([100]*cp.shape[0]), c=orig_colors, marker='*')
        ax1[1].set_ylim(-3, 20)
        ax2[1].scatter(points, o4, s=s, c=colors, norm=normalize)
        ax2[1].axhline(y=0, color='black', linestyle='--')
        ax2[1].scatter(cp2, np.array([-1.5]*cp2.shape[0]), s=np.array([100]*cp2.shape[0]), c=orig_colors, marker='*')
        ax2[1].set_ylim(-3, 20)
        # ax1[0].set_xlabel("point location", fontsize=16)
        # ax1[1].set_xlabel("point location", fontsize=16)
        # ax1.set_ylabel("Configuration 1 - weight")
        # ax2.set_ylabel("Configuration 2 - weight")
        ax1[0].set_ylabel("weight", fontsize=24)

    else:
        plt.scatter(points, offsets, s=s, c=colors, norm=normalize)
        plt.axhline(y=0, color='black', linestyle='--')
        plt.xlabel("point location")
        if do_ylim:
            plt.scatter(cp, np.array([-0.1]*cp.shape[0]), s=np.array([100]*cp.shape[0]), c=orig_colors, marker='*')
            plt.ylim(-0.2, 4)
            plt.ylabel("weight")
        else:
            plt.scatter(cp, t_cp, s=np.array([100]*cp.shape[0]), c=t_cp, marker='*', norm=normalize)
            plt.ylim(-6, 6)
            plt.ylabel("deformation (offest)")
    plt.savefig(filename)
    plt.show()

def deform_with_MLS(cage, new_cage, source_shape, cage_weights=None, pp_distances=None, alpha=0.5, deform_type='affine', epsilon=1e-8): 
    '''
    compute Moving Least Squares - Affine transformation.
    - ref: https://people.engr.tamu.edu/schaefer/research/mls.pdf
    Args:
    @Param 'cage': control points - (B, P, 3) 
    @Param 'new_cage': shifted control points - (B, P, 3) 
    @Param 'source_shape': source shape point cloud - (B, N, 3) 
    @Param 'cage_weights': control points weights  - (B, P) 
    @Param 'pp_distances': control points distances per point  - (B, N, P) 
    Outputs:
    @Param 'deformed_shape': deformed source shape point cloud - (B, N, 3)
    @Param 'w': un-normalized weights per point per control point - (B, N, P)
    @Param 'dist_w': un-normalized weights per point per control point based only on distance - (B, N, P)
    '''
    # some uniform casting
    cage = cage.float()
    new_cage = new_cage.float()
    source_shape = source_shape.float()

    # calculate the weights w_i for each point
    ss_exp = source_shape.unsqueeze(2) #(B, N, 1, 3)
    c_exp = cage.unsqueeze(1) #(B, 1, P, 3)
    nc_exp = new_cage.unsqueeze(1) #(B, 1, P, 3)
    norms = (ss_exp-c_exp)**2 #(B, N, P, 3)
    dists = torch.sum(norms, dim=3) ** alpha #(B, N, P)
    w = 1.0 / (dists+epsilon)  #(B, N, P)

    # w = 1.0 / (dists)  #(B, N, P)
    dist_w = w.clone()  #(B, N, P)
    if pp_distances is not None:
        w = pp_distances # (B, N, P) - each control point has its own weight
    if cage_weights is not None:
        w = w * cage_weights.unsqueeze(1) # (B, N, P) - each control point has its own weight
    # calculate p^* and q^*
    sum_w = torch.sum(w, dim=2, keepdim=True) #(B, N, 1)
    p_star = torch.matmul(w, cage) / (sum_w + epsilon) #(B, N, 3)
    q_star = torch.matmul(w, new_cage) / (sum_w + epsilon) #(B, N, 3)
    # calculate p^ and q^
    p_hat = c_exp - p_star.unsqueeze(2) #(B, N, P, 3)
    q_hat = nc_exp - q_star.unsqueeze(2) #(B, N, P, 3)
    if deform_type == 'affine':
        # calculate f_a
        def f_a(source_points):
            l = (source_points - p_star).unsqueeze(2) #(B, N, 1, 3) - left multiplier 
            p_hat_w = p_hat.transpose(3,2) * w.unsqueeze(2) #(B, N, 3, P)
            r = torch.matmul(p_hat_w, q_hat) #(B, N, 3, 3) - right multiplier
            problem = False
            try:
                pTwp = torch.matmul(p_hat_w, p_hat) #(B, N, 3, 3)
                m = torch.inverse(pTwp) #(B, N, 3, 3) - middle multiplier
            except:
                problem = True
                det = torch.det(pTwp) #(B, N, 1)
                good_idx = torch.where(torch.abs(det)>=1e-8)
                bad_idx = torch.where(torch.abs(det)<1e-8)
                good_pTwp = pTwp[good_idx[0], good_idx[1], :, :].unsqueeze(0) # TODO: adjust to batch bigger than 1
                m = torch.inverse(good_pTwp) #(B, N_g, 3, 3) - middle multiplier
                l = (source_points[good_idx[0], good_idx[1], ...] - p_star[good_idx[0], good_idx[1], ...]).unsqueeze(0).unsqueeze(2) #(B, N_g, 1, 3)
                r = torch.matmul(p_hat_w[good_idx[0], good_idx[1], ...].unsqueeze(0), q_hat[good_idx[0], good_idx[1], ...].unsqueeze(0)) #(B, N_g, 3, 3)
            new_points = torch.zeros(source_points.shape, dtype=source_points.dtype, device=cage.device) # (B, N, 3)
            if not problem: # all points has reversible matrix
                M = torch.matmul(m, r)
                new_points = torch.matmul(l, M).squeeze(2) + q_star # (B, N, 3)
            elif m.shape[1] > 0: # some of the points has singular matrix
                temp = torch.matmul(l, torch.matmul(m, r)).squeeze(2) + q_star[good_idx[0], good_idx[1], ...].unsqueeze(0) # (B, N_g, 3)
                new_points[good_idx[0], good_idx[1]] = temp
            if problem: # correct points where pTwp is singular
                new_points[bad_idx[0], bad_idx[1]] = source_points[bad_idx[0], bad_idx[1]] + q_star[bad_idx[0], bad_idx[1]] - p_star[bad_idx[0], bad_idx[1]]
            if torch.isnan(new_points).any():
                print("Encounter a nan in the deformed shape!\nThe cage is:\n{}\n the new cage is:\n{}\n".format(cage, new_cage))

            return new_points
    elif deform_type == 'rigid':
        # calculate f_r
        def f_r(source_points):     
            p_hat_w = p_hat.transpose(3,2) * w.unsqueeze(2) #(B, N, 3, P)
            PQt = torch.matmul(p_hat_w, q_hat) #(B, N, 3, 3)
            U, S, Vh = torch.linalg.svd(PQt) #(B, N, 3, 3), (B, N, 3), (B, N, 3, 3)
            M = torch.matmul(Vh.transpose(3,2), U.transpose(3,2)) # (B, N, 3, 3)
            r = (source_points - p_star).unsqueeze(-1) #(B, N, 3, 1)
            Mr = torch.matmul(M, r).squeeze(-1)
            new_points = Mr + q_star # (B, N, 3)

            return new_points

    # get deformed shape
    if deform_type == 'affine':
        deformed_shape = f_a(source_shape) # (B, N, 3)
    elif deform_type == 'rigid':
        deformed_shape = f_r(source_shape) # (B, N, 3)

    return deformed_shape, w, dist_w

def group_knn(k, query, points, unique=True, NCHW=True):
    """
    group batch of points to neighborhoods
    :param
        k: neighborhood size
        query: BxCxM or BxMxC
        points: BxCxN or BxNxC
        unique: neighborhood contains *unique* points
        NCHW: if true, the second dimension is the channel dimension
    :return
        neighbor_points BxCxMxk (if NCHW) or BxMxkxC (otherwise)
        index_batch     BxMxk
        distance_batch  BxMxk
    """
    if NCHW:
        batch_size, channels, num_points = points.size()
        points_trans = points.transpose(2, 1).contiguous()
        query_trans = query.transpose(2, 1).contiguous()
    else:
        points_trans = points.contiguous()
        query_trans = query.contiguous()

    batch_size, num_points, _ = points_trans.size()
    assert(num_points >= k
           ), "points size must be greater or equal to k"

    D = __batch_distance_matrix_general(query_trans, points_trans)
    if unique:
        # prepare duplicate entries
        points_np = points_trans.detach().cpu().numpy()
        indices_duplicated = np.ones(
            (batch_size, 1, num_points), dtype=np.int32)

        for idx in range(batch_size):
            _, indices = np.unique(points_np[idx], return_index=True, axis=0)
            indices_duplicated[idx, :, indices] = 0

        indices_duplicated = torch.from_numpy(
            indices_duplicated).to(device=D.device, dtype=torch.float32)
        D += torch.max(D) * indices_duplicated

    # (B,M,k)
    distances, point_indices = torch.topk(-D, k, dim=-1, sorted=True)
    # (B,N,C)->(B,M,N,C), (B,M,k)->(B,M,k,C)
    knn_trans = torch.gather(points_trans.unsqueeze(1).expand(-1, query_trans.size(1), -1, -1),
                             2,
                             point_indices.unsqueeze(-1).expand(-1, -1, -1, points_trans.size(-1)))

    if NCHW:
        knn_trans = knn_trans.permute(0, 3, 1, 2)

    return knn_trans, point_indices, -distances

def __batch_distance_matrix_general(A, B):
    """
    :param
        A, B [B,N,C], [B,M,C]
    :return
        D [B,N,M]
    """
    r_A = torch.sum(A * A, dim=2, keepdim=True)
    r_B = torch.sum(B * B, dim=2, keepdim=True)
    m = torch.matmul(A, B.permute(0, 2, 1))
    D = r_A - 2 * m + r_B.permute(0, 2, 1)
    return D

def dot_product(tensor1, tensor2, dim=-1, keepdim=False):
    return torch.sum(tensor1*tensor2, dim=dim, keepdim=keepdim)

def build_dataloader(opt):
    if opt.dataset == "SINGLE_SHAPE_TEST":
        dataset = SingleShapeDataset(opt)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True, num_workers=2)
    
    return dataloader

def log_outputs(opt, step, network, all_inputs, save_dir=None, save_all=False, KPDoutputs=None):
    if save_dir is None:
        save_dir = opt.log_dir
    faces_exists = 'source_face' in all_inputs and all_inputs['source_face'] is not None

    cp = all_inputs["source_cp"]
    device = cp.device 
    source_shape = all_inputs["source_shape"].to(device)
    deformed_cp = all_inputs["source_cp"] + all_inputs["offsets"][:, :cp.shape[1]].to(device)

    weights = network(source_shape.detach())
    weights = weights / opt.T
    weights = torch.softmax(weights, dim=-1)

    deformed_NeuralMLS, _, _ = deform_with_MLS(cp, deformed_cp, source_shape, None, weights, deform_type=opt.MLS_deform)
    deformed_MLS, _, _ = deform_with_MLS(cp, deformed_cp, source_shape, None, None, alpha=opt.MLS_alpha, deform_type=opt.MLS_deform)
    if save_all:
        if KPDoutputs is not None:
            deformed_KPD, _ = deform_with_KeypointDeformer(cp, deformed_cp, source_shape, KPDoutputs)
        run_arap = False
        if faces_exists:
            try:
                deformed_arap = deform_with_ARAP(cp, deformed_cp, source_shape, all_inputs['source_face'])
                run_arap = True
            except RuntimeError as e:                
                print("Failed to run ARAP:\n{}".format(e))

    s_fn = all_inputs['source_file'][0]
    s_shape = all_inputs["source_shape"][0].cpu()
    s_color = np.array([0.84, 0.37, 0]) # Source
    d_color = np.array([0, 0.45, 0.7]) # NeuralMLS
    m_color = np.array([0.5, 0.7, 0.2]) # MLS
    k_color = np.array([0.4, 0, 0.8]) # KPD
    a_color = np.array([0.7, 0.2, 0.5]) # ARAP
    cp_color = np.array([0.8, 0, 0.8]) # Control points
    tp_color = np.array([0.2, 0.2, 0.2]) # Target points

    if faces_exists:
        save_pts(os.path.join(save_dir,"step-{:06d}-shape-{}-Sa.obj".format(step, s_fn)), s_shape, colors=s_color, faces=all_inputs['source_face'][0].cpu())
        save_pts(os.path.join(save_dir,"step-{:06d}-shape-{}-Sab.obj".format(step, s_fn)), deformed_NeuralMLS[0].cpu(), colors=d_color, faces=all_inputs['source_face'][0].cpu())
    else:
        save_pts(os.path.join(save_dir,"step-{:06d}-shape-{}-Sa.obj".format(step, s_fn)), s_shape, colors=s_color)
        save_pts(os.path.join(save_dir,"step-{:06d}-shape-{}-Sab.obj".format(step, s_fn)), deformed_NeuralMLS[0].cpu(), colors=d_color)

    if save_all:
        if faces_exists:
            save_pts(os.path.join(save_dir,"shape-{}-Sab_CLASSIC.obj".format(s_fn)), deformed_MLS[0].cpu(), colors=m_color, faces=all_inputs['source_face'][0].cpu())
            if KPDoutputs is not None:
                save_pts(os.path.join(save_dir,"shape-{}-Sab_KPD.obj".format(s_fn)), deformed_KPD[0].cpu(), colors=k_color, faces=all_inputs['source_face'][0].cpu())
            if run_arap:
                save_pts(os.path.join(save_dir,"shape-{}-Sab_ARAP.obj".format(s_fn)), deformed_arap, colors=a_color, faces=all_inputs['source_face'][0].cpu())
        else:
            save_pts(os.path.join(save_dir,"shape-{}-Sab_CLASSIC.obj".format(s_fn)), deformed_MLS[0].cpu(), colors=m_color)
            if KPDoutputs is not None:
                save_pts(os.path.join(save_dir,"shape-{}-Sab_KPD.obj".format(s_fn)), deformed_KPD[0].cpu(), colors=k_color, faces=all_inputs['source_face'][0].cpu())

    save_pts(os.path.join(save_dir,"step-{:06d}-control_points.obj".format(step)), all_inputs["source_cp"][0].cpu(), colors=cp_color)
    save_pts(os.path.join(save_dir,"step-{:06d}-deformed_control_points.obj".format(step)), all_inputs["source_cp"][0].cpu() + all_inputs["offsets"][0, :cp.shape[1]].cpu(), colors=tp_color)
    
def deform_with_MLS_2D(cage, new_cage, source_shape, cage_weights=None, pp_distances=None, alpha=0.5, non_zero_axis=[0, 2], deform_type="affine"): 
    '''
    compute Moving Least Squares - Affine transformation.
    - ref: https://people.engr.tamu.edu/schaefer/research/mls.pdf
    Args:
    @Param 'cage': control points - (B, P, 3) 
    @Param 'new_cage': shifted control points - (B, P, 3) 
    @Param 'source_shape': source shape point cloud - (B, N, 3) 
    @Param 'cage_weights': control points weights  - (B, P) 
    @Param 'pp_distances': control points distances per point  - (B, N, P) 
    Outputs:
    @Param 'deformed_shape': deformed source shape point cloud - (B, N, 3)
    @Param 'w': un-normalized weights per point per control point - (B, N, P)
    @Param 'dist_w': un-normalized weights per point per control point based only on distance - (B, N, P)
    '''
    # some uniform casting
    cage = cage.float()
    new_cage = new_cage.float()
    source_shape = source_shape.float()
    # get rid of redundant 3D
    cage = cage[:,:, non_zero_axis]
    new_cage = new_cage[:,:, non_zero_axis]
    source_shape = source_shape[:,:, non_zero_axis]

    # calculate the weights w_i for each point
    ss_exp = source_shape.unsqueeze(2) #(B, N, 1, 2)
    c_exp = cage.unsqueeze(1) #(B, 1, P, 2)
    nc_exp = new_cage.unsqueeze(1) #(B, 1, P, 2)
    norms = (ss_exp-c_exp)**2 #(B, N, P, 2)
    dists = torch.sum(norms, dim=3) ** alpha #(B, N, P)
    w = 1.0 / (dists+1e-3)  #(B, N, P)
    dist_w = w.clone()  #(B, N, P)
    if pp_distances is not None:
        w = pp_distances # (B, N, P) - each control point has its own weight
    if cage_weights is not None:
        w = w * cage_weights.unsqueeze(1) # (B, N, P) - each control point has its own weight
    # calculate p^* and q^*
    sum_w = torch.sum(w, dim=2, keepdim=True) #(B, N, 1)
    p_star = torch.matmul(w, cage) / (sum_w + 1e-5) #(B, N, 2)
    q_star = torch.matmul(w, new_cage) / (sum_w + 1e-5) #(B, N, 2)
    # calculate p^ and q^
    p_hat = c_exp - p_star.unsqueeze(2) #(B, N, P, 2)
    q_hat = nc_exp - q_star.unsqueeze(2) #(B, N, P, 2)
    # calculate p^_perp and q^_perp
    p_hat_perp = torch.cat((-p_hat[:,:,:, 1].unsqueeze(-1), p_hat[:,:,:,0].unsqueeze(-1)), dim=-1) #(B, N, P, 2)
    q_hat_perp = torch.cat((-q_hat[:,:,:, 1].unsqueeze(-1), q_hat[:,:,:,0].unsqueeze(-1)), dim=-1) #(B, N, P, 2)

    if deform_type == "similarity":
        # calculate f_s
        def f_s(source_points):
            l = (source_points - p_star).unsqueeze(2) #(B, N, 1, 2) - left multiplier 

            mu_s = torch.sum(torch.sum((p_hat ** 2) * w.unsqueeze(-1), dim=-1, keepdim=True), dim=-2, keepdim=True) #(B, N, 1, 1)
            temp1 = torch.cat((p_hat.unsqueeze(-2), -p_hat_perp.unsqueeze(-2)), dim=-2) #(B, N, P, 2, 2)
            temp2 = torch.cat((q_hat.unsqueeze(-2), -q_hat_perp.unsqueeze(-2)), dim=-2).transpose(4,3) #(B, N, P, 2, 2)
            m = torch.matmul(temp1, temp2) * w.unsqueeze(-1).unsqueeze(-1) #(B, N, P, 2, 2)
            M = (1/mu_s) * torch.sum(m, dim=2) #(B, N, 2, 2)

            new_points = torch.matmul(l, M).squeeze(2) + q_star # (B, N, 2)

            return new_points    

    elif deform_type == "rigid":
        # calculate f_r
        def f_r(source_points):
            l = (source_points - p_star).unsqueeze(2) #(B, N, 1, 2) - left multiplier 

            mu_r1 = torch.sum(torch.sum((p_hat * q_hat) * w.unsqueeze(-1), dim=-1, keepdim=True), dim=-2, keepdim=True) #(B, N, 1, 1)
            mu_r2 = torch.sum(torch.sum((p_hat_perp * q_hat) * w.unsqueeze(-1), dim=-1, keepdim=True), dim=-2, keepdim=True) #(B, N, 1, 1)
            mu_r = torch.sqrt(mu_r1**2 + mu_r2**2) #(B, N, 1, 1

            temp1 = torch.cat((p_hat.unsqueeze(-2), -p_hat_perp.unsqueeze(-2)), dim=-2) #(B, N, P, 2, 2)
            temp2 = torch.cat((q_hat.unsqueeze(-2), -q_hat_perp.unsqueeze(-2)), dim=-2).transpose(4,3) #(B, N, P, 2, 2)
            m = torch.matmul(temp1, temp2) * w.unsqueeze(-1).unsqueeze(-1) #(B, N, P, 2, 2)
            M = (1/mu_r) * torch.sum(m, dim=2) #(B, N, 2, 2)

            new_points = torch.matmul(l, M).squeeze(2) + q_star # (B, N, 2)

            return new_points    

    else:
        # calculate f_a    
        def f_a(source_points):
            l = (source_points - p_star).unsqueeze(2) #(B, N, 1, 2) - left multiplier 
            p_hat_w = p_hat.transpose(3,2) * w.unsqueeze(2) #(B, N, 2, P)
            r = torch.matmul(p_hat_w, q_hat) #(B, N, 2, 2) - right multiplier
            problem = False
            try:
                pTwp = torch.matmul(p_hat_w, p_hat) #(B, N, 2, 2)
                m = torch.inverse(pTwp) #(B, N, 2, 2) - middle multiplier
            except:
                problem = True
                det = torch.det(pTwp) #(B, N, 1)
                good_idx = torch.where(torch.abs(det)>=1e-8)
                bad_idx = torch.where(torch.abs(det)<1e-8)
                good_pTwp = pTwp[good_idx[0], good_idx[1], :, :].unsqueeze(0) # TODO: adjust to batch bigger than 1
                m = torch.inverse(good_pTwp) #(B, N_g, 2, 2) - middle multiplier
                l = (source_points[good_idx[0], good_idx[1], ...] - p_star[good_idx[0], good_idx[1], ...]).unsqueeze(0).unsqueeze(2) #(B, N_g, 1, 2)
                r = torch.matmul(p_hat_w[good_idx[0], good_idx[1], ...].unsqueeze(0), q_hat[good_idx[0], good_idx[1], ...].unsqueeze(0)) #(B, N_g, 2, 2)
            new_points = torch.zeros(source_points.shape, dtype=source_points.dtype, device=cage.device) # (B, N, 2)
            if not problem: # all points has reversible matrix
                M = torch.matmul(m, r)
                new_points = torch.matmul(l, M).squeeze(2) + q_star # (B, N, 2)
            elif m.shape[1] > 0: # some of the points has singular matrix
                temp = torch.matmul(l, torch.matmul(m, r)).squeeze(2) + q_star[good_idx[0], good_idx[1], ...].unsqueeze(0) # (B, N_g, 2)
                new_points[good_idx[0], good_idx[1]] = temp
            if problem: # correct points where pTwp is singular
                new_points[bad_idx[0], bad_idx[1]] = source_points[bad_idx[0], bad_idx[1]] + q_star[bad_idx[0], bad_idx[1]] - p_star[bad_idx[0], bad_idx[1]]
            if torch.isnan(new_points).any():
                print("Encounter a nan in the deformed shape!\nThe cage is:\n{}\n the new cage is:\n{}\n".format(cage, new_cage))

            return new_points

    # get deformed shape
    if deform_type == "similarity":
        deformed_shape = f_s(source_shape) # (B, N, 2)
    elif deform_type == "rigid":
        deformed_shape = f_r(source_shape) # (B, N, 2)
    else:
        deformed_shape = f_a(source_shape) # (B, N, 2)

    B, N, _ = source_shape.shape
    deformed_shape_tmp = torch.zeros((B, N, 3), dtype=source_shape.dtype, device=source_shape.device)
    deformed_shape_tmp[:,:,non_zero_axis] = deformed_shape
    deformed_shape = deformed_shape_tmp

    return deformed_shape, w, dist_w

def deform_with_MLS_1D(cage, new_cage, source_shape, pp_distances=None, alpha=0.5, eps=1e-3): 
    '''
    compute Moving Least Squares - Affine transformation.
    - ref: https://people.engr.tamu.edu/schaefer/research/mls.pdf
    Args:
    @Param 'cage': control points - (P, 1) 
    @Param 'new_cage': shifted control points - (P, 1) 
    @Param 'source_shape': source shape point cloud - (N, 1) 
    @Param 'pp_distances': control points distances per point  - (N, P) 
    Outputs:
    @Param 'deformed_shape': deformed source shape point cloud - (N, 1)
    @Param 'w': un-normalized weights per point per control point - (N, P)
    @Param 'dist_w': un-normalized weights per point per control point based only on distance - (N, P)
    '''
    # some uniform casting
    cage = cage.float()
    new_cage = new_cage.float()
    source_shape = source_shape.float()

    # calculate the weights w_i for each point
    ss_exp = source_shape.unsqueeze(1) #(N, 1, 1)
    c_exp = cage.unsqueeze(0) #(1, P, 1)
    nc_exp = new_cage.unsqueeze(0) #(1, P, 1)
    norms = torch.abs(ss_exp-c_exp).squeeze(-1) #(N, P)
    dists = norms ** alpha #(N, P)
    w = 1.0 / (dists+eps)  #(N, P)
    dist_w = w.clone()  #(N, P)
    if pp_distances is not None:
        w = pp_distances # (N, P) - each control point has its own weight
    # calculate p^* and q^*
    sum_w = torch.sum(w, dim=1, keepdim=True) #(N, 1)
    p_star = torch.matmul(w, cage) / (sum_w + 1e-5) #(N, 1)
    q_star = torch.matmul(w, new_cage) / (sum_w + 1e-5) #(N, 1)
    # calculate p^ and q^
    p_hat = (c_exp - p_star.unsqueeze(1)).squeeze(-1) #(N, P)
    q_hat = (nc_exp - q_star.unsqueeze(1)).squeeze(-1) #(N, P)

    def f_a(source_points):
        l = (source_points - p_star) #(N, 1) - left multiplier 
        p_hat_w = p_hat * w #(N, P)
        r = torch.sum(p_hat_w * q_hat, dim=-1, keepdim=True) #(N, 1) - right multiplier
        pTwp = torch.sum(p_hat_w * p_hat, dim=-1, keepdim=True) #(N, 1)
        m = 1. / (pTwp+1e-8) #(N, 1) - middle multiplier
        M = m * r
        new_points = l * M + q_star # (N, 1)

        return new_points

    # get deformed shape
    deformed_shape = f_a(source_shape) # (N, 1)

    return deformed_shape, w, dist_w

def deform_with_ARAP(const, new_loc, vertices, faces):
    # find NN for each constraint and get its index
    v = np.array(vertices[0], dtype=np.float32)
    f = np.array(faces[0], dtype=np.int32)
    _, knn_idx, _ = group_knn(1, const, vertices, NCHW=False) # (B, P, K, D), (B, P, K)
    b = np.array(knn_idx[0, :, 0], dtype=np.int32)
    print("Init pre compute!!")
    L, M = robust_laplacian.mesh_laplacian(v, f)
    # M_inv = spla.inv(M)
    # strong_L = M_inv @ L
    deformer = ARAP(v.transpose(), f, b, anchor_weight=10, L=L)
    # deformer = ARAP(v.transpose(), f, b, anchor_weight=10, L=strong_L)
    print("Done pre compute!!")
    bc = np.array(new_loc[0], dtype=np.float32)
    anchors = dict(zip(b, bc))
    deformed = deformer(anchors, num_iters=10)
    deformed = deformed.transpose()
    deformed = torch.from_numpy(deformed)

    return deformed

def deform_with_KeypointDeformer(cp, deformed_cp, source_shape, KPDoutputs):
    B, _, _ = source_shape.shape

    cage = KPDoutputs['cage']
    cage_f = KPDoutputs['cage_face']
    influence = KPDoutputs['influence']

    base_cage = cage
    keypoints_offset = deformed_cp.transpose(1, 2) - cp.transpose(1, 2)
    cage_offset = torch.sum(keypoints_offset[..., None] * influence[:, None], dim=2)
    new_cage = base_cage + cage_offset.transpose(1, 2)

    # new_cage = new_cage.transpose(1, 2)
    deformed_shapes, weights, _ = deform_with_MVC(
        cage, new_cage, cage_f.expand(B, -1, -1), source_shape, verbose=True)

    return deformed_shapes, weights    