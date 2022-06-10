import torch
import torch.nn as nn
import numpy as np

from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d


def MLP(channels, batch_norm=True):
    if batch_norm:
        mlp = [Seq(Lin(channels[i - 1], channels[i]), ReLU(), BatchNorm1d(channels[i])) for i in range(1, len(channels)-1)]
        mlp += [Lin(channels[-2], channels[-1])]
        return Seq(*mlp)
    else:
        mlp = [Seq(Lin(channels[i - 1], channels[i]), ReLU()) for i in range(1, len(channels)-1)]
        mlp += [Lin(channels[-2], channels[-1])]
        return Seq(*mlp)

class WeightNet(nn.Module):
    def __init__(self, opt, max_iter=32, in_dim=3):
        super(WeightNet, self).__init__()
        self.opt = opt
        self.en_pos_enc = opt.en_pos_enc

        self.num_segs = opt.num_cp

        if opt.en_pos_enc:
            embed_fns = []
            embed_fns.append(lambda x: x)
            extra_embed_features = 0
            num_freq = 32
            sigma = opt.PE_sigma
            self.B_mat = torch.from_numpy(np.random.normal(loc=0.0, scale=sigma, size=(in_dim, num_freq)))        
            for fn in [torch.sin, torch.cos]:
                embed_fns.append(lambda x, fn=fn: fn(2*np.pi * torch.matmul(x.double(), self.B_mat.to(x.device))))
                extra_embed_features += num_freq
            self.embed = lambda inputs: torch.cat([fn(inputs) for fn in embed_fns], dim=-1)

            ### new adaptive embeddings ###
            if opt.phase == "train":
                self.adpative_mask = torch.zeros((in_dim + num_freq*2,))
                self.adpative_mask[:in_dim] = 1
                self.update_index = in_dim
                self.t = 0
                self.num_freq = num_freq
                self.jump = int((max_iter/2) / (num_freq/8)) # 16/4 = 4

            # self.mlp = MLP([in_dim+extra_embed_features, 256, 512, self.num_segs], batch_norm=False)
            self.mlp = MLP([in_dim+extra_embed_features, 1024, 1024, self.num_segs], batch_norm=False)

        else:        
            self.B_mat = None
            # self.mlp = MLP([in_dim, 1024, 1024, self.num_segs], batch_norm=False)
            self.mlp = MLP([in_dim, 128, 128, self.num_segs], batch_norm=False)

    def load_B_mat(self, B_mat):
        self.B_mat = B_mat
        embed_fns = []
        # for new adaptive encoding
        embed_fns.append(lambda x: x)

        for fn in [torch.sin, torch.cos]:
            embed_fns.append(lambda x, fn=fn: fn(2*np.pi * torch.matmul(x.double(), self.B_mat.to(x.device))))

        self.embed = lambda inputs: torch.cat([fn(inputs) for fn in embed_fns], dim=-1)

    def forward(self, point):
        if self.en_pos_enc:
            pos_enc = self.embed(point)
            if self.opt.phase == "train":
                # for new adaptive encoding
                # update enconding
                if self.update_index < len(self.adpative_mask) and self.t >= self.jump*2:
                    self.adpative_mask[self.update_index : self.update_index+self.jump] = ((self.t % self.jump) + 1) / self.jump            
                    self.adpative_mask[self.update_index+self.num_freq : self.update_index+self.jump+self.num_freq] = ((self.t % self.jump) + 1) / self.jump            
                    if self.t % self.jump == self.jump - 1:
                        self.update_index += self.jump
                pos_enc = pos_enc * self.adpative_mask
                self.t += 1
        else:
            pos_enc = point

        weights = self.mlp(pos_enc.float())

        return weights