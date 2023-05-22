import numpy as np
import time
from sklearn.metrics import f1_score, precision_score, recall_score
import bottleneck as bn

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

# consider delay threshold and missing segments
def get_range_proba(predict, label, delay=7):
    splits = np.where(label[1:] != label[:-1])[0] + 1
    is_anomaly = label[0] == 1
    new_predict = np.array(predict)
    pos = 0

    for sp in splits:
        if is_anomaly:
            if 1 in predict[pos:min(pos + delay + 1, sp)]:
                new_predict[pos: sp] = 1
            else:
                new_predict[pos: sp] = 0
        is_anomaly = not is_anomaly
        pos = sp
    sp = len(label)

    if is_anomaly:  # anomaly in the end
        if 1 in predict[pos: min(pos + delay + 1, sp)]:
            new_predict[pos: sp] = 1
        else:
            new_predict[pos: sp] = 0

    return new_predict


# set missing = 0
def reconstruct_label(timestamp, label):
    timestamp = np.asarray(timestamp, np.int64)
    index = np.argsort(timestamp)

    timestamp_sorted = np.asarray(timestamp[index])
    interval = np.min(np.diff(timestamp_sorted))

    label = np.asarray(label, np.int64)
    label = np.asarray(label[index])

    idx = (timestamp_sorted - timestamp_sorted[0]) // interval

    new_label = np.zeros(shape=((timestamp_sorted[-1] - timestamp_sorted[0]) // interval + 1,), dtype=np.int)
    new_label[idx] = label

    return new_label


def eval_ad_result(test_pred_list, test_labels_list, test_timestamps_list, delay):
    labels = []
    pred = []
    for test_pred, test_labels, test_timestamps in zip(test_pred_list, test_labels_list, test_timestamps_list):
        assert test_pred.shape == test_labels.shape == test_timestamps.shape
        test_labels = reconstruct_label(test_timestamps, test_labels)
        test_pred = reconstruct_label(test_timestamps, test_pred)
        test_pred = get_range_proba(test_pred, test_labels, delay)
        labels.append(test_labels)
        pred.append(test_pred)
    labels = np.concatenate(labels)
    pred = np.concatenate(pred)
    return {
        'f1': f1_score(labels, pred),
        'precision': precision_score(labels, pred),
        'recall': recall_score(labels, pred)
    }


def np_shift(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result

def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def eval_anomaly_detect(model,all_train_data, all_train_labels,
                        all_test_data, all_test_labels, ESP=1E-5):
    train_length = all_train_data.shape[1]
    test_length = all_test_data.shape[1]
    full_repr = model.casual_encode(
        np.concatenate([all_train_data, all_test_data],axis = 1),
        mask='mask_last',
        casual=True,
        sliding_length=1,
        sliding_padding=100,
        batch_size=256
    ).squeeze()
    all_train_repr = full_repr[:train_length]
    all_test_repr = full_repr[-test_length:]

    full_repr_wom = model.casual_encode(
        np.concatenate([all_train_data, all_test_data],axis = 1),
        casual=True,
        sliding_length=1,
        sliding_padding=100,
        batch_size=256
    ).squeeze()
    # nan_mask = ~full_repr.isnan().any(axis=-1)
    # nan_mask_float = nan_mask.float()
    # full_repr_wom = full_repr_wom * nan_mask_float
    all_train_repr_wom = full_repr_wom[:train_length]
    all_test_repr_wom = full_repr_wom[-test_length:]

    train_err = np.abs(all_train_repr_wom - all_train_repr).sum(axis=1)
    test_err = np.abs(all_test_repr_wom - all_test_repr).sum(axis=1)

    # combined_energy = np.concatenate([train_err, test_err], axis=0)
    # threshold = np.percentile(combined_energy, 100 - 1)
    # print("Threshold :", threshold)

    ma = np_shift(bn.move_mean(np.concatenate([train_err, test_err]), 21), 1)
    train_err_adj = (train_err[22:] - ma[22:train_length]) / (ma[22:train_length]+ESP)
    test_err_adj = (test_err - ma[-test_length:]) / (ma[-test_length:]+ESP)
    #train_err_adj = train_err_adj[22:]

    thr = np.mean(train_err_adj) + 2 * np.std(train_err_adj)
    test_res = (test_err_adj > thr) * 1

    gt = all_test_labels.astype(int)
    pred = test_res
    gt, pred = adjustment(gt, test_res)

    pred = np.array(pred)
    gt = np.array(gt).reshape(pred.shape)

    print("pred: ", pred.shape)
    print("gt:   ", gt.shape)

    accuracy = accuracy_score(gt, pred)
    f1  = f1_score(gt, pred)
    P  = precision_score(gt, pred)
    R  = recall_score(gt, pred)
    print(f1,P,R)
    precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
    print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
        accuracy, precision,
        recall, f_score))
    return accuracy, precision, recall, f_score

def eval_anomaly_detect1(model,all_train_data, all_train_labels,
                        all_test_data, all_test_labels,anomaly_ratio=25, ESP=1E-5):
    train_length = all_train_data.shape[1]
    test_length = all_test_data.shape[1]
    full_repr = model.casual_encode(
        np.concatenate([all_train_data, all_test_data],axis = 1),
        mask='mask_last',
        casual=True,
        sliding_length=1,
        sliding_padding=100,
        batch_size=256
    ).squeeze()
    all_train_repr = full_repr[:train_length]
    all_test_repr = full_repr[-test_length:]

    full_repr_wom = model.casual_encode(
        np.concatenate([all_train_data, all_test_data],axis = 1),
        casual=True,
        sliding_length=1,
        sliding_padding=100,
        batch_size=256
    ).squeeze()
    # nan_mask = ~full_repr.isnan().any(axis=-1)
    # nan_mask_float = nan_mask.float()
    # full_repr_wom = full_repr_wom * nan_mask_float
    all_train_repr_wom = full_repr_wom[:train_length]
    all_test_repr_wom = full_repr_wom[-test_length:]

    train_err = np.abs(all_train_repr_wom - all_train_repr).sum(axis=1)
    test_err = np.abs(all_test_repr_wom - all_test_repr).sum(axis=1)

    # combined_energy = np.concatenate([train_err, test_err], axis=0)
    # threshold = np.percentile(combined_energy, 100 - 1)
    # print("Threshold :", threshold)
    combined_energy = np.concatenate([train_err, test_err], axis=0)
    threshold = np.percentile(combined_energy, 100 - anomaly_ratio)

    # ma = np_shift(bn.move_mean(np.concatenate([train_err, test_err]), 21), 1)
    # train_err_adj = (train_err[22:] - ma[22:train_length]) / (ma[22:train_length]+ESP)
    # test_err_adj = (test_err - ma[-test_length:]) / (ma[-test_length:]+ESP)
    #train_err_adj = train_err_adj[22:]

    # thr = np.mean(train_err_adj) + 2 * np.std(train_err_adj)
    test_res = (test_err > threshold) * 1

    gt = all_test_labels.astype(int)
    pred = test_res
    gt, pred = adjustment(gt, test_res)

    pred = np.array(pred)
    gt = np.array(gt).reshape(pred.shape)

    print("pred: ", pred.shape)
    print("gt:   ", gt.shape)

    accuracy = accuracy_score(gt, pred)
    f1  = f1_score(gt, pred)
    P  = precision_score(gt, pred)
    R  = recall_score(gt, pred)
    print(f1,P,R)
    precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
    print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
        accuracy, precision,
        recall, f_score))
    return accuracy, precision, recall, f_score

def eval_anomaly_detection(model, all_train_data, all_train_labels, all_train_timestamps,
                                  all_test_data, all_test_labels, all_test_timestamps, delay):
    t = time.time()
    
    all_train_repr = {}
    all_test_repr = {}
    all_train_repr_wom = {}
    all_test_repr_wom = {}
    for k in all_train_data:
        train_data = all_train_data[k]
        test_data = all_test_data[k]

        full_repr = model.encode(
            np.concatenate([train_data, test_data]).reshape(1, -1, 1),
            mask='mask_last',
            casual=True,
            sliding_length=1,
            sliding_padding=200,
            batch_size=256
        ).squeeze()

        all_train_repr[k] = full_repr[:len(train_data)]
        all_test_repr[k] = full_repr[len(train_data):]

        full_repr_wom = model.encode(
            np.concatenate([train_data, test_data]).reshape(1, -1, 1),
            casual=True,
            sliding_length=1,
            sliding_padding=200,
            batch_size=256
        ).squeeze()
        all_train_repr_wom[k] = full_repr_wom[:len(train_data)]
        all_test_repr_wom[k] = full_repr_wom[len(train_data):]
        
    res_log = []
    labels_log = []
    timestamps_log = []
    for k in all_train_data:
        train_data = all_train_data[k]
        train_labels = all_train_labels[k]
        train_timestamps = all_train_timestamps[k]

        test_data = all_test_data[k]
        test_labels = all_test_labels[k]
        test_timestamps = all_test_timestamps[k]

        train_err = np.abs(all_train_repr_wom[k] - all_train_repr[k]).sum(axis=1)
        test_err = np.abs(all_test_repr_wom[k] - all_test_repr[k]).sum(axis=1)

        ma = np_shift(bn.move_mean(np.concatenate([train_err, test_err]), 21), 1)
        train_err_adj = (train_err - ma[:len(train_err)]) / ma[:len(train_err)]
        test_err_adj = (test_err - ma[len(train_err):]) / ma[len(train_err):]
        train_err_adj = train_err_adj[22:]

        thr = np.mean(train_err_adj) + 4 * np.std(train_err_adj)
        test_res = (test_err_adj > thr) * 1

        for i in range(len(test_res)):
            if i >= delay and test_res[i-delay:i].sum() >= 1:
                test_res[i] = 0

        res_log.append(test_res)
        labels_log.append(test_labels)
        timestamps_log.append(test_timestamps)
    t = time.time() - t
    
    eval_res = eval_ad_result(res_log, labels_log, timestamps_log, delay)
    eval_res['infer_time'] = t
    return res_log, eval_res


def eval_anomaly_detection_coldstart(model, all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay):
    t = time.time()
    
    all_data = {}
    all_repr = {}
    all_repr_wom = {}
    for k in all_train_data:
        all_data[k] = np.concatenate([all_train_data[k], all_test_data[k]])
        all_repr[k] = model.encode(
            all_data[k].reshape(1, -1, 1),
            mask='mask_last',
            casual=True,
            sliding_length=1,
            sliding_padding=200,
            batch_size=256
        ).squeeze()
        all_repr_wom[k] = model.encode(
            all_data[k].reshape(1, -1, 1),
            casual=True,
            sliding_length=1,
            sliding_padding=200,
            batch_size=256
        ).squeeze()
        
    res_log = []
    labels_log = []
    timestamps_log = []
    for k in all_data:
        data = all_data[k]
        labels = np.concatenate([all_train_labels[k], all_test_labels[k]])
        timestamps = np.concatenate([all_train_timestamps[k], all_test_timestamps[k]])
        
        err = np.abs(all_repr_wom[k] - all_repr[k]).sum(axis=1)
        ma = np_shift(bn.move_mean(err, 21), 1)
        err_adj = (err - ma) / ma
        
        MIN_WINDOW = len(data) // 10
        thr = bn.move_mean(err_adj, len(err_adj), MIN_WINDOW) + 4 * bn.move_std(err_adj, len(err_adj), MIN_WINDOW)
        res = (err_adj > thr) * 1
        
        for i in range(len(res)):
            if i >= delay and res[i-delay:i].sum() >= 1:
                res[i] = 0

        res_log.append(res[MIN_WINDOW:])
        labels_log.append(labels[MIN_WINDOW:])
        timestamps_log.append(timestamps[MIN_WINDOW:])
    t = time.time() - t
    
    eval_res = eval_ad_result(res_log, labels_log, timestamps_log, delay)
    eval_res['infer_time'] = t
    return res_log, eval_res

