import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from torch.func import functional_call, stack_module_state


class Ensemble(nn.Module):
    """
    Vectorized ensemble of modules.
    """

    def __init__(self, modules, **kwargs):
        super().__init__()
        # combine_state_for_ensemble causes graph breaks
        # Use torch.func utilities to stack module parameters for vectorized evaluation.
        params, buffers = stack_module_state(modules)
        self.params = params
        self.buffers = buffers
        self.module = deepcopy(modules[0])
        self._repr = str(modules[0])
        self._n = len(modules)

    def __len__(self):
        return self._n

    def _slice_state(self, idx):
        params = {k: v[idx] for k, v in self.params.items()}
        buffers = {k: v[idx] for k, v in self.buffers.items()}
        return params, buffers

    def _call(self, params, buffers, *args, **kwargs):
        return functional_call(self.module, (params, buffers), args, kwargs)

    def forward(self, *args, **kwargs):
        outputs = []
        for i in range(self._n):
            params, buffers = self._slice_state(i)
            outputs.append(self._call(params, buffers, *args, **kwargs))
        return torch.stack(outputs)

    def __repr__(self):
        return f'Vectorized {len(self)}x ' + self._repr


class ShiftAug(nn.Module):
    """
    Random shift image augmentation.
    Adapted from https://github.com/facebookresearch/drqv2
    """
    def __init__(self, pad=3):
        super().__init__()
        self.pad = pad
        self.padding = tuple([self.pad] * 4)

    def forward(self, x):
        x = x.float()
        n, _, h, w = x.size()
        assert h == w
        x = F.pad(x, self.padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
        shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)
        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)


class PixelPreprocess(nn.Module):
    """
    Normalizes pixel observations to [-0.5, 0.5].
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.div(255.).sub(0.5)


class SimNorm(nn.Module):
    """
    Simplicial normalization.
    Adapted from https://arxiv.org/abs/2204.00616.
    """

    def __init__(self, cfg):
        super().__init__()
        self.dim = cfg.simnorm_dim

    def forward(self, x):
        shp = x.shape
        x = x.view(*shp[:-1], -1, self.dim)
        x = F.softmax(x, dim=-1)
        return x.view(*shp)

    def __repr__(self):
        return f"SimNorm(dim={self.dim})"


class NormedLinear(nn.Linear):
    """
    Linear layer with LayerNorm, activation, and optionally dropout.
    """

    def __init__(self, *args, dropout=0., act=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ln = nn.LayerNorm(self.out_features)
        if act is None:
            act = nn.Mish(inplace=False)
        self.act = act
        self.dropout = nn.Dropout(dropout, inplace=False) if dropout else None

    def forward(self, x):
        x = super().forward(x)
        if self.dropout:
            x = self.dropout(x)
        return self.act(self.ln(x))

    def __repr__(self):
        repr_dropout = f", dropout={self.dropout.p}" if self.dropout else ""
        return f"NormedLinear(in_features={self.in_features}, "\
            f"out_features={self.out_features}, "\
            f"bias={self.bias is not None}{repr_dropout}, "\
            f"act={self.act.__class__.__name__})"


def mlp(in_dim, mlp_dims, out_dim, act=None, dropout=0.):
    """
    Basic building block of TD-MPC2.
    MLP with LayerNorm, Mish activations, and optionally dropout.
    """
    if isinstance(mlp_dims, int):
        mlp_dims = [mlp_dims]
    dims = [in_dim] + mlp_dims + [out_dim]
    mlp = nn.ModuleList()
    for i in range(len(dims) - 2):
        mlp.append(NormedLinear(dims[i], dims[i+1], dropout=dropout*(i==0)))
    mlp.append(NormedLinear(dims[-2], dims[-1], act=act) if act else nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*mlp)


def conv(in_shape, num_channels, act=None):
    """
    Basic convolutional encoder for TD-MPC2 with raw image observations.
    4 layers of convolution with ReLU activations, followed by a linear layer.
    """
    assert in_shape[-1] == 64 # assumes rgb observations to be 64x64
    layers = [
        ShiftAug(), PixelPreprocess(),
        nn.Conv2d(in_shape[0], num_channels, 7, stride=2), nn.ReLU(inplace=False),
        nn.Conv2d(num_channels, num_channels, 5, stride=2), nn.ReLU(inplace=False),
        nn.Conv2d(num_channels, num_channels, 3, stride=2), nn.ReLU(inplace=False),
        nn.Conv2d(num_channels, num_channels, 3, stride=1), nn.Flatten()]
    if act:
        layers.append(act)
    return nn.Sequential(*layers)


def enc(cfg, out={}):
    """
    Returns a dictionary of encoders for each observation in the dict.
    """
    obs_shape = cfg.obs_shape
    obs_type = getattr(cfg, "obs_type", "states")
    if isinstance(obs_shape, str):
        obs_dim_val = getattr(cfg, "obs_dim", 0)
        obs_dim = int(obs_dim_val) if str(obs_dim_val).isdigit() else 0
        obs_shape = {obs_type: (obs_dim,)}
        cfg.obs_shape = obs_shape
    if not isinstance(obs_shape, dict):
        obs_shape = {obs_type: tuple(obs_shape)}
        cfg.obs_shape = obs_shape

    for k in obs_shape.keys():
        if k in ("state", "states", "proprio", "obs"):
            out[k] = mlp(
                obs_shape[k][0] + cfg.task_dim,
                max(cfg.num_enc_layers - 1, 1) * [cfg.enc_dim],
                cfg.latent_dim,
                act=SimNorm(cfg),
            )
        elif k in ("rgb", "pixels"):
            out[k] = conv(obs_shape[k], cfg.num_channels, act=SimNorm(cfg))
        else:
            out[k] = mlp(
                obs_shape[k][0] + cfg.task_dim,
                max(cfg.num_enc_layers - 1, 1) * [cfg.enc_dim],
                cfg.latent_dim,
                act=SimNorm(cfg),
            )
    return nn.ModuleDict(out)


def api_model_conversion(target_state_dict, source_state_dict):
    """
    Converts a checkpoint from our old API to the new torch.compile compatible API.

    NOTE:
    In some older checkpoints, tensordict metadata (e.g., ``__batch_size`` or
    ``__device``) can appear in parameter names. These metadata entries are not
    present in the current model's ``state_dict`` and should not be treated as
    learnable parameters during conversion. The conversion therefore only copies
    keys that exist in the target model and skips metadata-style keys.
    """
    # check whether checkpoint is already in the new format
    if "_detach_Qs_params.0.weight" in source_state_dict:
        return source_state_dict

    name_map = ['weight', 'bias', 'ln.weight', 'ln.bias']
    new_state_dict = target_state_dict.copy()

    def is_metadata(key: str) -> bool:
        return "__batch_size" in key or "__device" in key

    # rename keys
    for key, val in list(source_state_dict.items()):
        if key.startswith('_Qs.'):
            num = key[len('_Qs.params.'):]
            new_key = str(int(num) // 4) + "." + name_map[int(num) % 4]
            for prefix in ("_Qs.params.", "_detach_Qs_params."):
                candidate_key = prefix + new_key
                if candidate_key in target_state_dict and not is_metadata(candidate_key):
                    new_state_dict[candidate_key] = val
            del source_state_dict[key]
        elif key.startswith('_target_Qs.'):
            num = key[len('_target_Qs.params.'):]
            new_key = str(int(num) // 4) + "." + name_map[int(num) % 4]
            candidate_key = "_target_Qs_params." + new_key
            if candidate_key in target_state_dict and not is_metadata(candidate_key):
                new_state_dict[candidate_key] = val
            del source_state_dict[key]

    # copy remaining parameters that directly match the target model
    for key, val in source_state_dict.items():
        if key in target_state_dict and not is_metadata(key):
            new_state_dict[key] = val

    return new_state_dict
