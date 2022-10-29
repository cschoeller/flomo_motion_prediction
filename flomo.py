import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from .spline_flow import NeuralSplineFlow, FCN


class RNN(nn.Module):
    """ GRU based recurrent neural network. """

    def __init__(self, nin, nout, es=16, hs=16, nl=3, device=0):
        super().__init__()
        self.embedding = nn.Linear(nin, es)
        self.gru = nn.GRU(input_size=es, hidden_size=hs, num_layers=nl, batch_first=True)
        self.output_layer = nn.Linear(hs, nout)
        self.device = device
        self.cuda(self.device)

    def forward(self, x, hidden=None):
        x = F.relu(self.embedding(x))
        x, hidden = self.gru(x, hidden)
        x = self.output_layer(x)
        return x, hidden

class FloMo(nn.Module):

    def __init__(self, hist_size, pred_steps, alpha, beta, gamma, B=15., momentum=0.2,
                use_map=False, interactions=False, rel_coords=True, norm_rotation=False, device=0):
        super().__init__()
        self.pred_steps = pred_steps
        self.output_size = pred_steps * 2
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.B = B
        self.rel_coords = rel_coords
        self.norm_rotation = norm_rotation

        # core modules
        self.obs_encoding_size = 16
        self.obs_encoder = RNN(nin=2, nout=self.obs_encoding_size, device=device)
        self.flow = NeuralSplineFlow(nin=self.output_size, nc=self.obs_encoding_size, n_layers=10, K=8,
                                    B=self.B, hidden_dim=[32, 32, 32, 32, 32], device=device)

        # move model to specified device
        self.device = device
        self.to(device)

    def _encode_conditionals(self, x):
        # encode observed trajectory
        x_in = x
        if self.rel_coords:
            x_in = x[:,1:] - x[:,:-1] # convert to relative coords
        x_enc, hidden = self.obs_encoder(x_in) # encode relative histories
        x_enc = x_enc[:,-1]
        x_enc_context = x_enc
        return x_enc_context

    def _abs_to_rel(self, y, x_t):
        y_rel = y - x_t # future trajectory relative to x_t
        y_rel[:,1:] = (y_rel[:,1:] - y_rel[:,:-1]) # steps relative to each other
        y_rel = y_rel * self.alpha # scale up for numeric reasons
        return y_rel

    def _rel_to_abs(self, y_rel, x_t):
        y_abs = y_rel / self.alpha
        return torch.cumsum(y_abs, dim=-2) + x_t 

    def _rotate(self, x, x_t, angles_rad):
        c, s = torch.cos(angles_rad), torch.sin(angles_rad)
        c, s = c.unsqueeze(1), s.unsqueeze(1)
        x_center = x - x_t # translate
        x_vals, y_vals = x_center[:,:,0], x_center[:,:,1]
        new_x_vals = c * x_vals + (-1 * s) * y_vals # _rotate x
        new_y_vals = s * x_vals + c * y_vals # _rotate y
        x_center[:,:,0] = new_x_vals
        x_center[:,:,1] = new_y_vals
        return x_center + x_t # translate back

    def _normalize_rotation(self, x, y_true=None):
        x_t = x[:,-1:,:]
        # compute rotation angle, such that last timestep aligned with (1,0)
        x_t_rel = x[:,-1] - x[:,-2]
        rot_angles_rad = -1 * torch.atan2(x_t_rel[:,1], x_t_rel[:,0])
        x = self._rotate(x, x_t, rot_angles_rad)
        
        if y_true != None:
            y_true = self._rotate(y_true, x_t, rot_angles_rad)
            return x, y_true, rot_angles_rad # inverse

        return x, rot_angles_rad # forward pass

    def _inverse(self, y_true, x):
        if self.norm_rotation:
            x, y_true, angle = self._normalize_rotation(x, y_true)

        x_t = x[...,-1:,:]
        x_enc = self._encode_conditionals(x) # history encoding
        y_rel = self._abs_to_rel(y_true, x_t)
        y_rel_flat = torch.flatten(y_rel, start_dim=1)

        if self.training:
            # add noise to zero values to avoid infinite density points
            zero_mask = torch.abs(y_rel_flat) < 1e-2 # approx. zero
            noise = torch.randn_like(y_rel_flat) * self.beta
            y_rel_flat = y_rel_flat + (zero_mask * noise)

            # minimally perturb remaining motion to avoid x1 = x2 for any values
            noise = torch.randn_like(y_rel_flat) * self.gamma
            y_rel_flat = y_rel_flat + (~zero_mask * noise)
        
        z, jacobian_det = self.flow.inverse(torch.flatten(y_rel_flat, start_dim=1), x_enc)
        return z, jacobian_det

    def _repeat_rowwise(self, c_enc, n):
        org_dim = c_enc.size(-1)
        c_enc = c_enc.repeat(1, n)
        return c_enc.view(-1, n, org_dim)

    def forward(self, z, c):
        raise NotImplementedError

    def sample(self, n, x):
        with torch.no_grad():
            if self.norm_rotation:
                x, rot_angles_rad = self._normalize_rotation(x)
            x_enc = self._encode_conditionals(x) # history encoding
            x_enc_expanded = self._repeat_rowwise(x_enc, n).view(-1, self.obs_encoding_size)
            n_total = n * x.size(0)
            output_shape = (x.size(0), n, self.pred_steps, 2) # predict n trajectories input

            # sample and compute likelihoods
            z = torch.randn([n_total, self.output_size]).to(self.device)
            samples_rel, log_det = self.flow(z, x_enc_expanded)
            samples_rel = samples_rel.view(*output_shape)
            normal = Normal(0, 1, validate_args=True)
            log_probs = (normal.log_prob(z).sum(1) - log_det).view((x.size(0), -1))

            x_t = x[...,-1:,:].unsqueeze(dim=1).repeat(1, n, 1, 1)
            samples_abs = self._rel_to_abs(samples_rel, x_t)

            # invert rotation normalization
            if self.norm_rotation:
                x_t_all = x[...,-1,:]
                for i in range(len(samples_abs)):
                    pred_trajs = samples_abs[i]
                    samples_abs[i] = self._rotate(pred_trajs, x_t_all[i], -1 * rot_angles_rad[i].repeat(pred_trajs.size(0)))

            return samples_abs, log_probs

    def log_prob(self, y_true, x):
        z, log_abs_jacobian_det = self._inverse(y_true, x)
        normal = Normal(0, 1, validate_args=True)
        return normal.log_prob(z).sum(1) + log_abs_jacobian_det
