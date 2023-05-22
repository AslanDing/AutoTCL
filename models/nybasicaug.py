import numpy as np
from utils import take_per_row, split_with_nan, centerize_vary_length_series, torch_pad_nan
import numpy as np
from tqdm import tqdm
import utils as hlp
import torch

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


def rotation(x):
    flip = np.random.choice([-1, 1], size=(x.shape[0], x.shape[2]))
    rotate_axis = np.arange(x.shape[2])
    np.random.shuffle(rotate_axis)
    flip = flip[:, np.newaxis, :]
    flip_tensor = torch.as_tensor(flip).cuda()
    res =  ( flip_tensor* x[:, :, rotate_axis])
    return res


def rotation_s(x, plot=False):
    flip = np.random.choice([-1], size=(1, x.shape[1]))
    rotate_axis = np.arange(x.shape[1])
    np.random.shuffle(rotate_axis)
    x_ = flip[:, :] * x[:, rotate_axis]
    if plot:
        hlp.plot1d(x, x_, save_file='aug_examples/rotation_s.png')
    return x_.astype(np.float32)


def rotation2d(x, sigma=0.2):
    thetas = np.random.normal(loc=0, scale=sigma, size=(x.shape[0]))
    c = np.cos(thetas)
    s = np.sin(thetas)

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        rot = np.array(((c[i], -s[i]), (s[i], c[i])))
        ret[i] = np.dot(pat, rot)
    return ret.astype(np.float32)


def permutation(x, max_segments=5, seg_mode="equal"):
    orig_steps = np.arange(x.shape[1])

    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[1] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[warp]
        else:
            ret[i] = pat
    return ret.astype(np.float32)


def magnitude_warp(x_torch, sigma=0.1, knot=4):
    x = x_torch.cpu().detach().numpy()
    from scipy.interpolate import CubicSpline
    orig_steps = np.arange(x.shape[1])

    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot + 2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2], 1)) * (np.linspace(0, x.shape[1] - 1., num=knot + 2))).T

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):

        li = []
        for dim in range(x.shape[2]):
            li.append(CubicSpline(warp_steps[:, dim], random_warps[i, :, dim])(orig_steps))
        warper = np.array(li).T

        ret[i] = pat * warper

    return totensor(ret.astype(np.float32))


def self_time_warp(x_torch, sigma=0.1, knot=4):
    x = x_torch.cpu().detach().numpy()
    from scipy.interpolate import CubicSpline
    orig_steps = np.arange(x.shape[1])

    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot + 2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2], 1)) * (np.linspace(0, x.shape[1] - 1., num=knot + 2))).T

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            time_warp = CubicSpline(warp_steps[:, dim], warp_steps[:, dim] * random_warps[i, :, dim])(orig_steps)
            scale = (x.shape[1] - 1) / time_warp[-1]
            ret[i, :, dim] = np.interp(orig_steps, np.clip(scale * time_warp, 0, x.shape[1] - 1), pat[:, dim]).T
    return totensor(ret.astype(np.float32))


def window_slice(x_torch, reduce_ratio=0.4):
    x = x_torch.cpu().detach().numpy()

    # https://halshs.archives-ouvertes.fr/halshs-01357973/document
    target_len = np.ceil(reduce_ratio * x.shape[1]).astype(int)
    if target_len >= x.shape[1]:
        return x
    starts = np.random.randint(low=0, high=x.shape[1] - target_len, size=(x.shape[0])).astype(int)
    ends = (target_len + starts).astype(int)

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            ret[i, :, dim] = np.interp(np.linspace(0, target_len, num=x.shape[1]), np.arange(target_len),
                                       pat[starts[i]:ends[i], dim]).T
    return totensor(ret.astype(np.float32))


def window_warp(x_torch, window_ratio=0.3, scales=[0.5, 2.]):
    x = x_torch.cpu().detach().numpy()

    # https://halshs.archives-ouvertes.fr/halshs-01357973/document
    warp_scales = np.random.choice(scales, x.shape[0])
    warp_size = np.ceil(window_ratio * x.shape[1]).astype(int)
    window_steps = np.arange(warp_size)

    window_starts = np.random.randint(low=1, high=x.shape[1] - warp_size - 1, size=(x.shape[0])).astype(int)
    window_ends = (window_starts + warp_size).astype(int)

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            start_seg = pat[:window_starts[i], dim]
            window_seg = np.interp(np.linspace(0, warp_size - 1, num=int(warp_size * warp_scales[i])), window_steps,
                                   pat[window_starts[i]:window_ends[i], dim])
            end_seg = pat[window_ends[i]:, dim]
            warped = np.concatenate((start_seg, window_seg, end_seg))
            ret[i, :, dim] = np.interp(np.arange(x.shape[1]), np.linspace(0, x.shape[1] - 1., num=warped.size),
                                       warped).T
    return totensor(ret.astype(np.float32))


