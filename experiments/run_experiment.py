import os
import datetime
import torch

from utils.utils import BaseOptions
from create_user_study import create_user_study
from mls_ablation import mls_ablation
from pws_ga_2D import create_2D_illustration_of_piecewise_smoothness_and_geometry_awareness
from pws_ga_1D import create_1D_illustration_of_piecewise_smoothness_and_geometry_awareness
from create_comparisons import create_comparisons
from visualizations.visualize import visualize, temperature_visualization


if __name__ == "__main__":
    parser = BaseOptions()
    opt = parser.parse()

    if opt.ckpt is not None:
        opt.log_dir = os.path.dirname(opt.ckpt)
    else:
        opt.prev_log_dir = opt.log_dir
        opt.log_dir = os.path.join(opt.log_dir, "-".join(filter(None, [os.path.basename(__file__)[:-3],
                                                         datetime.datetime.now().strftime("%d-%m-%Y__%H:%M:%S"),
                                                        opt.name]))) # log dir will be the file name + date_time + expirement name

    ### setup device ###
    if opt.not_cuda:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    opt.device = device

    if opt.phase == "visualize":
        visualize(opt, save_subdir=opt.subdir)
        
    elif opt.phase == 'create_comparisons':
        assert(opt.KPDckpt), "Can't create the same comparisons without a KPD pre-trained model"
        os.makedirs(opt.log_dir, exist_ok=True)
        create_comparisons(opt)

    elif opt.phase == 'create_temperature_illustration':
        temperature_visualization(opt, save_subdir=opt.subdir)
        
    elif opt.phase == "create_user_study":
        os.makedirs(opt.log_dir, exist_ok=True)
        create_user_study(opt)
        
    elif opt.phase == "mls_ablation":
        opt.log_dir = opt.prev_log_dir # for the ablations we want to aggreagate the experiments
        os.makedirs(opt.log_dir, exist_ok=True)
        mls_ablation(opt, opt.subdir)
        
    elif opt.phase == "create_2d_illustration":
        os.makedirs(opt.log_dir, exist_ok=True)
        create_2D_illustration_of_piecewise_smoothness_and_geometry_awareness(opt)

    elif opt.phase == "create_1d_illustration":
        os.makedirs(opt.log_dir, exist_ok=True)
        create_1D_illustration_of_piecewise_smoothness_and_geometry_awareness(opt)



