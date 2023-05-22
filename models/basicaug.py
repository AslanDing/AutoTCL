import numpy as np
from utils import take_per_row, split_with_nan, centerize_vary_length_series, torch_pad_nan
import numpy as np
from tqdm import tqdm
import utils as hlp
import torch
import tsaug
import torch
import time
from torch.nn.functional import interpolate

def totensor(x):
    return torch.from_numpy(x).type(torch.FloatTensor).cuda()

def npcutout(x,temporal_unit=0):
    ts_l = x.size(1)
    crop_l = np.random.randint(low=2 ** (temporal_unit + 1), high=ts_l + 1)
    crop_left = np.random.randint(ts_l - crop_l + 1)
    crop_right = crop_left + crop_l
    crop_eleft = np.random.randint(crop_left + 1)
    crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
    crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0))

    a1 = take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft)
    a1 = a1[:, -crop_l:]

    a2 = take_per_row(x, crop_offset + crop_left, crop_eright - crop_left)
    a2 = a2[:, :crop_l]
    return a1, a2


def subsequence(x):
    ts = x
    seq_len = ts.shape[1]
    ts_l = x.size(1)
    crop_l = np.random.randint(low=2, high=ts_l + 1)
    new_ts = ts.clone()
    start = np.random.randint(ts_l - crop_l + 1)
    end = start + crop_l
    start = max(0, start)
    end = min(end, seq_len)
    new_ts[:,:start,:] = 0.0
    new_ts[:,end:,:] = 0.0
    return new_ts

def slidewindow(ts, horizon=.2, stride=0.2):
    xf = []
    yf = []
    for i in range(0, ts.shape[0], int(stride * ts.shape[0])):
        horizon1 = int(horizon * ts.shape[0])
        if (i + horizon1 + horizon1 <= ts.shape[0]):
            xf.append(ts[i:i + horizon1, 0])
            yf.append(ts[i + horizon1:i + horizon1 + horizon1, 0])

    xf = np.asarray(xf)
    yf = np.asarray(yf)

    return xf, yf


def cutout(ts, perc=0.1):
    seq_len = ts.shape[1]
    new_ts = ts.clone()
    win_len = int(perc * seq_len)
    start = np.random.randint(0, seq_len - win_len - 1)
    end = start + win_len
    start = max(0, start)
    end = min(end, seq_len)
    # print("[INFO] start={}, end={}".format(start, end))
    new_ts[:,start:end,:] = 0.0
    # return new_ts, ts[start:end, ...]
    return new_ts

def jitter(x, sigma=0.3):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + torch.normal(mean=0., std=sigma, size=x.shape).cuda()


def scaling(x, sigma=0.5):
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = torch.normal(mean=1., std=sigma, size=(x.shape[0], x.shape[2])).cuda()
    res = torch.multiply(x, torch.unsqueeze(factor,1))
    return res

warp_transform = tsaug.TimeWarp(n_speed_change=100, max_speed_ratio=10)
def magnitude_warp(x_torch):
    x = x_torch.cpu().detach().numpy()
    x_t = np.transpose(x, (0, 2, 1))
    x_tran = warp_transform.augment(x_t).transpose((0, 2, 1))
    return totensor(x_tran.astype(np.float32))


def time_warp(x_torch):
    x = x_torch.cpu().detach().numpy()
    x_tran =  warp_transform.transform.augment(x)
    return totensor(x_tran.astype(np.float32))

def window_slice(x_torch, reduce_ratio=0.4):
    x = torch.transpose(x_torch, 2, 1)

    target_len = np.ceil(reduce_ratio * x.shape[2]).astype(int)
    if target_len >= x.shape[2]:
        return x
    starts = np.random.randint(low=0, high=x.shape[2] - target_len, size=(x.shape[0])).astype(int)
    ends = (target_len + starts).astype(int)
    croped_x = torch.stack([x[i, :, starts[i]:ends[i]] for i in range(x.shape[0])], 0)

    ret = interpolate(croped_x, x.shape[2], mode='linear', align_corners=False)
    ret = torch.transpose(ret, 2, 1)
    return ret


def window_warp(x_torch, window_ratio=0.3, scales=[0.5, 2.]):
    B, T, D = x_torch.size()
    x = torch.transpose(x_torch, 2, 1)
    # https://halshs.archives-ouvertes.fr/halshs-01357973/document
    warp_scales = np.random.choice(scales, B)
    warp_size = np.ceil(window_ratio * T).astype(int)

    window_starts = np.random.randint(low=1, high=T - warp_size - 1, size=(B)).astype(int)
    window_ends = (window_starts + warp_size).astype(int)

    rets = []

    for i in range(x.shape[0]):
        window_seg = torch.unsqueeze(x[i, :, window_starts[i]:window_ends[i]], 0)
        window_seg_inter = interpolate(window_seg, int(warp_size * warp_scales[i]), mode='linear', align_corners=False)[
            0]
        start_seg = x[i, :, :window_starts[i]]
        end_seg = x[i, :, window_ends[i]:]
        ret_i = torch.cat([start_seg, window_seg_inter, end_seg], -1)
        ret_i_inter = interpolate(torch.unsqueeze(ret_i, 0), T, mode='linear', align_corners=False)
        rets.append(ret_i_inter)

    ret = torch.cat(rets, 0)
    ret = torch.transpose(ret, 2, 1)
    return ret

