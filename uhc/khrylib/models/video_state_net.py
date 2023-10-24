import torch.nn as nn
from uhc.khrylib.models.tcn import TemporalConvNet
from uhc.khrylib.models.rnn import RNN
from uhc.khrylib.utils.torch import *


class VideoStateNet(nn.Module):
    def __init__(self, cnn_feat_dim, v_hdim=128, v_margin=10, v_net_type='lstm', v_net_param=None, causal=False):
        super().__init__()
        self.mode = 'test'
        self.cnn_feat_dim = cnn_feat_dim
        self.v_net_type = v_net_type
        self.v_hdim = v_hdim
        self.v_margin = v_margin
        if v_net_type == 'lstm':
            self.v_net = RNN(cnn_feat_dim, v_hdim, v_net_type, bi_dir=not causal)
        elif v_net_type == 'tcn':
            if v_net_param is None:
                v_net_param = {}
            tcn_size = v_net_param.get('size', [64, 128])
            dropout = v_net_param.get('dropout', 0.2)
            kernel_size = v_net_param.get('kernel_size', 3)
            assert tcn_size[-1] == v_hdim
            self.v_net = TemporalConvNet(cnn_feat_dim, tcn_size, kernel_size=kernel_size, dropout=dropout, causal=causal)
        self.v_out = None
        self.t = 0
        # training only
        self.indices = None
        self.scatter_indices = None
        self.gather_indices = None
        self.cnn_feat_ctx = None

    def set_mode(self, mode):
        self.mode = mode

    def initialize(self, x):
        if self.mode == 'test':
            self.v_out = self.forward_v_net(x.unsqueeze(1)).squeeze(1)[self.v_margin:-self.v_margin]
            self.t = 0
        elif self.mode == 'train':
            masks, cnn_feat, v_metas = x
            device, dtype = masks.device, masks.dtype
            end_indice = np.where(masks.cpu().numpy() == 0)[0]
            v_metas = v_metas[end_indice, :]
            num_episode = len(end_indice)
            end_indice = np.insert(end_indice, 0, -1)
            max_episode_len = int(np.diff(end_indice).max())
            self.indices = np.arange(masks.shape[0])
            for i in range(1, num_episode):
                start_index = end_indice[i] + 1
                end_index = end_indice[i + 1] + 1
                self.indices[start_index:end_index] += i * max_episode_len - start_index
            self.cnn_feat_ctx = np.zeros((max_episode_len + 2*self.v_margin, num_episode, self.cnn_feat_dim))
            for i in range(num_episode):
                exp_ind, start_ind = v_metas[i, :]
                self.cnn_feat_ctx[:, i, :] = cnn_feat[exp_ind][start_ind - self.v_margin: start_ind + max_episode_len + self.v_margin]
            self.cnn_feat_ctx = tensor(self.cnn_feat_ctx, dtype=dtype, device=device)
            self.scatter_indices = LongTensor(np.tile(self.indices[:, None], (1, self.cnn_feat_dim))).to(device)
            self.gather_indices = LongTensor(np.tile(self.indices[:, None], (1, self.v_hdim))).to(device)

    def forward(self, x):
        if self.mode == 'test':
            x = torch.cat((self.v_out[[self.t], :], x), dim=1)
            self.t += 1
        elif self.mode == 'train':
            v_ctx = self.forward_v_net(self.cnn_feat_ctx)[self.v_margin:-self.v_margin]
            v_ctx = v_ctx.transpose(0, 1).contiguous().view(-1, self.v_hdim)
            v_out = torch.gather(v_ctx, 0, self.gather_indices)
            x = torch.cat((v_out, x), dim=1)
        return x

    def forward_v_net(self, x):
        if self.v_net_type == 'tcn':
            x = x.permute(1, 2, 0).contiguous()
        x = self.v_net(x)
        if self.v_net_type == 'tcn':
            x = x.permute(2, 0, 1).contiguous()
        return x


if __name__ == '__main__':
    import pickle
    import os

    dtype = torch.float64
    torch.set_default_dtype(dtype)
    takes = ['1205_take_01', '1205_take_03']
    cnn_feat_dict, _ = pickle.load(open(os.path.expanduser('~/datasets/egopose/features/cnn_feat_0111.p'), 'rb'))
    cnn_feat = [cnn_feat_dict[x] for x in takes]
    net = VideoStateNet(cnn_feat[0].shape[-1], v_net_type='tcn')
    net.initialize(tensor(cnn_feat[0], dtype=dtype))
    action = net(ones((1, 32)))
    print(action.shape)
