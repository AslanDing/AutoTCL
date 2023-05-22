import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from .dilated_conv_down import DilatedConvEncoder as DilatedConvEncoder_down
from .dilated_conv import DilatedConvEncoder


def generate_continuous_mask(B, T, n=5, l=0.1):
    res = torch.full((B, T), True, dtype=torch.bool)
    if isinstance(n, float):
        n = int(n * T)
    n = max(min(n, T // 2), 1)
    
    if isinstance(l, float):
        l = int(l * T)
    l = max(l, 1)
    
    for i in range(B):
        for _ in range(n):
            t = np.random.randint(T-l+1)
            res[i, t:t+l] = False
    return res

def generate_binomial_mask(B, T, p=0.5):
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)

class TSEncoder(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=64, depth=10, mask_mode='binomial',dropout=0.1,bias_init=0.5):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims

        if self.output_dims == 1:
            init_tensor = torch.from_numpy(bias_init*np.ones([1,output_dims])).float()
            self.bias = nn.Parameter(data=init_tensor,
                                     requires_grad=True)

        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],
            kernel_size=3
        )
        self.repr_dropout = None if dropout==0.0 else nn.Dropout(p=dropout)
        
    def forward(self, x, mask=None):  # x: B x T x input_dims

        nan_mask = ~x.isnan().any(axis=-1)
        nan_mask_float = nan_mask.float()
        x = x * nan_mask_float.unsqueeze(2)
        # x[~nan_mask] = 0
        x = self.input_fc(x)  # B x T x Ch

        # nan_mask = ~x.isnan().any(axis=-1)
        # x[~nan_mask] = 0
        # x = self.input_fc(x)  # B x T x Ch
        
        # generate & apply mask
        if mask is None:
            if self.training:
                mask = self.mask_mode
            else:
                mask = 'all_true'

        if mask == 'binomial':
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'continuous':
            mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'all_true':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == 'all_false':
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == 'mask_last':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False
        
        # mask &= nan_mask
        # x[~mask] = 0

        mask = mask.float()
        mask = mask * nan_mask_float
        x = x * mask.unsqueeze(2)

        # conv encoder
        x = x.transpose(1, 2)  # B x Ch x T
        x = self.feature_extractor(x)

        if self.repr_dropout is not None:
            x = self.repr_dropout(x)  # B x Co x T
        x = x.transpose(1, 2)  # B x T x Co

        if self.output_dims == 1:
            x = x + self.bias.view(1,1,1)

        return x


class DCNN(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=64, mask_mode='binomial',down_sampling=True):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.feature_extractor = DilatedConvEncoder_down(in_channels=hidden_dims,out_channels=output_dims)

        self.repr_dropout = nn.Dropout(p=0.1)

    def forward(self, x, mask=None,output_hiddens = False):  # x: B x T x input_dims
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0
        x = self.input_fc(x)  # B x T x Ch

        # generate & apply mask
        if mask is None:
            if self.training:
                mask = self.mask_mode
            else:
                mask = 'all_true'

        if mask == 'binomial':
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'continuous':
            mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'all_true':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == 'all_false':
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == 'mask_last':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False

        mask &= nan_mask
        x[~mask] = 0

        # conv encoder
        x = x.transpose(1, 2)  # B x Ch x T
        if output_hiddens:
            hiddens = self.feature_extractor.get_hiddens(x)
            dropout_hiddens = [self.repr_dropout(x)  for x in hiddens]# B x Co x T
            hiddens_t =  [x.transpose(1, 2)  for x in dropout_hiddens]# B x T x Co
            return hiddens_t
        else:
            x = self.feature_extractor(x)
            x = self.repr_dropout(x)  # B x Co x T
            x = x.transpose(1, 2)  # B x T x Co

        return x