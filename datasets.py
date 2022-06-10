import torch
import numpy as np
import os

from keypoint_deformer.keypointdeformer.utils.utils import normalize_to_box
from utils.data_utils import read_obj, read_off, get_control_points_for_shape


class SingleShapeDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self.source_files = opt.source_model
        self.phase = opt.phase
        self.normalize = True

        pref = os.path.commonpath([os.path.dirname(f) for f in self.source_files])
        self.source_names = [os.path.relpath(f, pref) for f in self.source_files]

        self.control_points, self.offsets = get_control_points_for_shape(self.source_files[0], self.phase) 
        assert self.control_points is not None, "the given input shape does not have a set of control points annotated on it" 

        if self.phase == 'create_user_study':
            self.target_points = self.offsets + self.control_points.unsqueeze(0)
        else:
            self.target_points = self.offsets + self.control_points 
            
        opt.num_cp = len(self.control_points)

    def __len__(self):
        return len(self.source_files)

    def __getitem__(self, idx):
        source_file = self.source_files[idx]
        source_face = None
        if source_file[-4:] == ".xyz":
            source_mesh = np.loadtxt(source_file)
            source_face = None
        elif source_file[-4:] == ".off":
            source_mesh, source_face = read_off(source_file) 
            source_mesh, source_face = np.array(source_mesh), np.array(source_face)
        elif source_file[-4:] == ".obj":
            source_mesh, source_face = read_obj(source_file) 
            source_mesh, source_face = np.array(source_mesh), np.array(source_face)

        source_mesh = torch.from_numpy(source_mesh[:,:3].astype(np.float32))
        # Normalize
        center = 0
        scale = 1
        if self.normalize:
            source_mesh, center, scale = normalize_to_box(source_mesh)
            
        source_cp = (self.control_points - center) / scale
        target_cp = (self.target_points - center) / scale
        if self.phase == 'create_user_study':
            self.offsets = target_cp - source_cp.unsqueeze(0)
        else:
            self.offsets = target_cp - source_cp
        offsets = self.offsets

        source_fn = os.path.splitext(self.source_names[idx])[0].replace("/","_")

        if source_face is not None:
            return {"source_shape": source_mesh.squeeze(0), "source_cp": source_cp, "offsets": offsets, "source_file": source_fn, "target_cp": target_cp, "source_face": source_face}
        else:
            return {"source_shape": source_mesh.squeeze(0), "source_cp": source_cp, "offsets": offsets, "source_file": source_fn, "target_cp": target_cp}
