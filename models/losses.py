import numpy as np
import torch
import random
import torch.nn.functional as F

criterion = torch.nn.CrossEntropyLoss()

def hierarchical_contrastive_loss(z1, z2, alpha=0.5, temporal_unit=0):
    loss = torch.tensor(0., device=z1.device)
    d = 0
    while z1.size(1) > 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        if d >= temporal_unit:
            if 1 - alpha != 0:
                loss += (1 - alpha) * temporal_contrastive_loss(z1, z2)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
    if z1.size(1) == 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        d += 1
    return loss / d



def subsequence_loss(z1, z2, alpha=0.5, temporal_unit=0):
    loss = torch.tensor(0., device=z1.device)
    d = 0
    s1 = torch.unsqueeze(torch.max(z1, 1)[0], 1)
    s2 = torch.unsqueeze(torch.max(z2, 1)[0], 1)
    loss = instance_contrastive_loss(s1, s2)
    exit(1)
    return loss


def subsequence_infoNCE(z1, z2, pooling='max',temperature=1.0, k = 16):
    #   z1, z2    B X T X D
    B = z1.size(0)
    T = z1.size(1)
    D = z1.size(2)
    crop_size = int(T/k)
    crop_leng = crop_size*k

    # random start?
    start = random.randint(0,T-crop_leng)
    crop_z1 = z1[:,start:start+crop_leng,:]
    crop_z2 = z2[:,start:start+crop_leng,:]

    crop_z1 = crop_z1.view(B ,k,crop_size,D)
    crop_z2 = crop_z2.view(B ,k,crop_size,D)

    # debug
    # crop_z1 = crop_z1.reshape(B * k, crop_size, D)
    # crop_z2 = crop_z2.reshape(B * k, crop_size, D)
    # return instance_contrastive_loss(crop_z1, crop_z2)+temporal_contrastive_loss(crop_z1,crop_z2)


    if pooling=='max':
        # crop_z1_pooling = torch.max(crop_z1,2)[0]
        # crop_z2_pooling = torch.max(crop_z2,2)[0]
        # crop_z1_pooling = torch.unsqueeze(crop_z1_pooling.view(B*k, D), 1)
        # crop_z2_pooling = torch.unsqueeze(crop_z2_pooling.view(B*k, D), 1)


        crop_z1 = crop_z1.reshape(B*k,crop_size,D)
        crop_z2 = crop_z2.reshape(B*k,crop_size,D)

        crop_z1_pooling = F.max_pool1d(crop_z1.transpose(1, 2).contiguous(), kernel_size=crop_size).transpose(1, 2)
        crop_z2_pooling = F.max_pool1d(crop_z2.transpose(1, 2).contiguous(), kernel_size=crop_size).transpose(1, 2)

    elif pooling=='mean':
        crop_z1_pooling = torch.unsqueeze(torch.mean(z1,1),1)
        crop_z2_pooling = torch.unsqueeze(torch.mean(z2,1),1)


    return InfoNCE(crop_z1_pooling,crop_z2_pooling,temperature)


def sliding_local_infoNCE(z1, z2, pooling='max', temperature=1.0, k=16, sliding = 16, negtive_number = 16):
    B = z1.size(0)
    T = z1.size(1)
    D = z1.size(2)
    crop_leng = int(T / k)

    # pos
    start = random.randint(0, crop_leng)
    anchors = []
    poss = []
    negs = []
    while start < T - crop_leng:
        anchors.append(z1[:, start:start + crop_leng, :])

        pos_can_start = max(start-crop_leng,0)
        pos_can_end = min(start+crop_leng,T - crop_leng)

        pos_start = random.randint(pos_can_start, pos_can_end)
        poss.append(z1[:, pos_start:pos_start + crop_leng, :])

        neg = []
        # choose a negative one
        while len(neg)<negtive_number:
            start = random.randint(0, T - crop_leng)
            if start>=pos_can_start and start<=pos_can_end:
                continue
            neg.append(z1[:, start:start + crop_leng, :])
        neg_sample = torch.stack(neg,1)
        negs.append(neg_sample)
        start +=sliding

    anchors_array = torch.stack(anchors,0)
    poss_array = torch.stack(poss,0)
    negs_array = torch.stack(negs,0)

    if pooling=='max':
        anchors_array_pooling = torch.max(anchors_array,2)[0]
        poss_array_pooling = torch.max(poss_array,2)[0]
        negs_array_pooling = torch.max(negs_array,3)[0]
    elif pooling=='mean':
        anchors_array_pooling = torch.mean(anchors_array,2)
        poss_array_pooling = torch.mean(poss_array,2)
        negs_array_pooling = torch.mean(negs_array,3)

    # merge the first two dim
    anchors_array_pooling = anchors_array_pooling.reshape(anchors_array_pooling.shape[0]*anchors_array_pooling.shape[1],1,anchors_array_pooling.shape[2])
    poss_array_pooling = poss_array_pooling.reshape(poss_array_pooling.shape[0]*poss_array_pooling.shape[1],1,poss_array_pooling.shape[2])
    negs_array_pooling = negs_array_pooling.reshape(negs_array_pooling.shape[0]*negs_array_pooling.shape[1],negtive_number,negs_array_pooling.shape[-1])

    apn = torch.cat([anchors_array_pooling,poss_array_pooling,negs_array_pooling],1)

    apn_T = apn.transpose(1,2)

    # B X K * K
    similarity_matrices = torch.bmm(apn, apn_T)[:,1:,:]

    logits = similarity_matrices / temperature
    logits = -F.log_softmax(logits, dim=-1)
    loss = logits[:,0].mean()

    return loss


def local_infoNCE(z1, z2, pooling='max',temperature=1.0, k = 16):
    #   z1, z2    B X T X D
    z1 = torch.nn.functional.normalize(z1,dim=2)
    B = z1.size(0)
    T = z1.size(1)
    D = z1.size(2)
    crop_size = int(T/k)
    if crop_size<1:
        return 0
    crop_leng = crop_size*k

    start = random.randint(0,T-crop_leng)
    crop_z1 = z1[:,start:start+crop_leng,:]
    crop_z1 = crop_z1.view(B ,k,crop_size,D)

    if pooling=='max':
        crop_z1 = crop_z1.reshape(B*k,crop_size,D)
        crop_z1_pooling = F.max_pool1d(crop_z1.transpose(1, 2).contiguous(), kernel_size=crop_size).transpose(1, 2).reshape(B,k,D)

    elif pooling=='mean':
        crop_z1_pooling = torch.unsqueeze(torch.mean(z1,1),1)

    crop_z1_pooling_T = crop_z1_pooling.transpose(1,2)

    # B X K * K
    similarity_matrices = torch.bmm(crop_z1_pooling, crop_z1_pooling_T)

    labels = torch.eye(k-1, dtype=torch.float32)
    labels = torch.cat([labels,torch.zeros(1,k-1)],0)
    labels = torch.cat([torch.zeros(k,1),labels],-1)

    pos_labels = labels.cuda()
    pos_labels[k-1,k-2]=1.0


    neg_labels = labels.T + labels + torch.eye(k)
    neg_labels[0,2]=1.0
    neg_labels[-1,-3]=1.0
    neg_labels = neg_labels.cuda()


    similarity_matrix = similarity_matrices[0]

    # select and combine multiple positives
    positives = similarity_matrix[pos_labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~neg_labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)

    logits = logits / temperature
    logits = -F.log_softmax(logits, dim=-1)
    loss = logits[:,0].mean()

    return loss


def L1out(z1, z2, pooling='max',temperature=1.0):
    if pooling == 'max':
        z1 = F.max_pool1d(z1.transpose(1, 2).contiguous(), kernel_size=z1.size(1)).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2).contiguous(), kernel_size=z2.size(1)).transpose(1, 2)
    elif pooling == 'mean':
        z1 = torch.unsqueeze(torch.mean(z1, 1), 1)
        z2 = torch.unsqueeze(torch.mean(z2, 1), 1)

    batch_size = z1.size(0)

    features = torch.cat([z1, z2], dim=0).squeeze(1)  # 2B x T x C
    features = torch.nn.functional.normalize(features,dim=1)

    labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.cuda()

    similarity_matrix = torch.matmul(features, features.T)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)

    # logits = logits / temperature
    mmax = torch.unsqueeze(torch.max(logits,-1)[0],-1)
    neg_pos = negatives - mmax
    exp_negatives = torch.exp(neg_pos)
    sum_negs = torch.sum(exp_negatives,-1)+1e-6
    pos_exp = torch.exp(positives-mmax)+1e-5
    logits = -torch.log(pos_exp/sum_negs)
    loss = logits.mean()

    return loss


def global_infoNCE(z1, z2, pooling='max',temperature=1.0):
    if pooling == 'max':
        z1 = F.max_pool1d(z1.transpose(1, 2).contiguous(), kernel_size=z1.size(1)).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2).contiguous(), kernel_size=z2.size(1)).transpose(1, 2)
    elif pooling == 'mean':
        z1 = torch.unsqueeze(torch.mean(z1, 1), 1)
        z2 = torch.unsqueeze(torch.mean(z2, 1), 1)

    # return instance_contrastive_loss(z1, z2)
    return InfoNCE(z1,z2,temperature)

def InfoNCE_backup(z1, z2, temperature=1.0):

    batch_size = z1.size(0)

    features = torch.cat([z1, z2], dim=0).squeeze(1)  # 2B x T x C

    labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.cuda()

    similarity_matrix = torch.matmul(features, features.T)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)

    logits = logits / temperature
    logits = -F.log_softmax(logits, dim=-1)
    loss = logits[:,0].mean()

    return loss

def InfoNCE(z1, z2, temperature=1.0,l1out=False):

    assert z1.shape[0] == z2.shape[0]

    z1t = torch.nn.functional.normalize(z1,dim=2)
    z2t = torch.nn.functional.normalize(z2,dim=2)

    similarity_matrix = torch.matmul(z1t.squeeze(1), z2t.squeeze(1).T)

    mask = torch.eye(z1.shape[0], dtype=torch.bool).cuda()

    if not l1out:
        positives = similarity_matrix[mask].view(mask.shape[0], -1)
        negatives = similarity_matrix[~mask].view(mask.shape[0], mask.shape[1]-1)

        logits = torch.cat([positives, negatives], dim=1)

        logits = logits / temperature
        logits = -F.log_softmax(logits, dim=-1)
        loss = logits[:, 0].mean()
    else:
        positives = similarity_matrix[mask].view(mask.shape[0], -1)

        negatives = similarity_matrix[~mask].view(mask.shape[0], mask.shape[1]-1)
        exp_negatives = torch.exp(negatives)
        sum_negs = torch.sum(exp_negatives, -1) + 1e-6

        pos_exp = torch.exp(positives) + 1e-5
        logits = torch.log(pos_exp/(sum_negs/(mask.shape[1]-1)))
        loss = logits.mean()

    return loss

def infoNCE(z1, z2, pooling='max',temperature=1.0):
    if pooling == 'max':
        z1 = F.max_pool1d(z1.transpose(1, 2).contiguous(), kernel_size=z1.size(1)).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2).contiguous(), kernel_size=z2.size(1)).transpose(1, 2)
    elif pooling == 'mean':
        z1 = torch.unsqueeze(torch.mean(z1, 1), 1)
        z2 = torch.unsqueeze(torch.mean(z2, 1), 1)

    # return instance_contrastive_loss(z1, z2)
    return InfoNCE(z1,z2,temperature)

def l1Out(z1, z2, pooling='max',temperature=1.0):
    if pooling == 'max':
        z1 = F.max_pool1d(z1.transpose(1, 2).contiguous(), kernel_size=z1.size(1)).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2).contiguous(), kernel_size=z2.size(1)).transpose(1, 2)
    elif pooling == 'mean':
        z1 = torch.unsqueeze(torch.mean(z1, 1), 1)
        z2 = torch.unsqueeze(torch.mean(z2, 1), 1)

    # return instance_contrastive_loss(z1, z2)
    return InfoNCE(z1,z2,temperature,True)

def sim(z1, z2, pooling='max',temperature=1.0):
    if pooling == 'max':
        z1 = F.max_pool1d(z1.transpose(1, 2).contiguous(), kernel_size=z1.size(1)).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2).contiguous(), kernel_size=z2.size(1)).transpose(1, 2)
    elif pooling == 'mean':
        z1 = torch.unsqueeze(torch.mean(z1, 1), 1)
        z2 = torch.unsqueeze(torch.mean(z2, 1), 1)

    assert z1.shape[0] == z2.shape[0]

    z1t = torch.nn.functional.normalize(z1, dim=2)
    z2t = torch.nn.functional.normalize(z2, dim=2)

    similarity_matrix = torch.matmul(z1t.squeeze(1), z2t.squeeze(1).T)

    mask = torch.eye(z1.shape[0], dtype=torch.bool).cuda()
    positives = similarity_matrix[mask].view(mask.shape[0], -1)
    loss = positives[:, 0].mean()
    return loss

def instance_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=0)  # 2B x T x C
    z = z.transpose(0, 1)  # T x 2B x C
    sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
    # remove self-similarities
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # T x 2B x (2B-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    
    i = torch.arange(B, device=z1.device)
    loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
    return loss

def temporal_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=1)  # B x 2T x C
    sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # B x 2T x (2T-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    
    t = torch.arange(T, device=z1.device)
    loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
    return loss

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)#/len(kernel_val)

def mmdx(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None, pooling = "max"):
    if pooling == 'max':
        source = F.max_pool1d(source.transpose(1, 2).contiguous(), kernel_size=source.size(1)).transpose(1, 2)
        target = F.max_pool1d(target.transpose(1, 2).contiguous(), kernel_size=target.size(1)).transpose(1, 2)
    elif pooling == 'mean':
        source = torch.unsqueeze(torch.mean(source, 1), 1)
        target = torch.unsqueeze(torch.mean(target, 1), 1)

    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source.squeeze(1), target.squeeze(1),
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss

def mmd(source, target):

    source_mean = torch.mean(source,dim=0)
    target_mean = torch.mean(target,dim=0)

    result = torch.sum(torch.square(source_mean-target_mean))

    return result
