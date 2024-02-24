import os
import glob
import numpy as np
import torch
import pandas as pd
import math
import random
from datetime import datetime
import pickle
from scipy.io.arff import loadarff
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def load_UCR(dataset,use_fft=False):
    train_file = os.path.join('datasets/UCR', dataset, dataset + "_TRAIN.tsv")
    test_file = os.path.join('datasets/UCR', dataset, dataset + "_TEST.tsv")
    train_df = pd.read_csv(train_file, sep='\t', header=None)
    test_df = pd.read_csv(test_file, sep='\t', header=None)
    train_array = np.array(train_df)
    test_array = np.array(test_df)

    # Move the labels to {0, ..., L-1}
    labels = np.unique(train_array[:, 0])
    transform = {}
    for i, l in enumerate(labels):
        transform[l] = i

    train = train_array[:, 1:].astype(np.float64)
    train_labels = np.vectorize(transform.get)(train_array[:, 0])
    test = test_array[:, 1:].astype(np.float64)
    test_labels = np.vectorize(transform.get)(test_array[:, 0])

    # Normalization for non-normalized datasets
    # To keep the amplitude information, we do not normalize values over
    # individual time series, but on the whole dataset
    if dataset in [
        'AllGestureWiimoteX',
        'AllGestureWiimoteY',
        'AllGestureWiimoteZ',
        'BME',
        'Chinatown',
        'Crop',
        'EOGHorizontalSignal',
        'EOGVerticalSignal',
        'Fungi',
        'GestureMidAirD1',
        'GestureMidAirD2',
        'GestureMidAirD3',
        'GesturePebbleZ1',
        'GesturePebbleZ2',
        'GunPointAgeSpan',
        'GunPointMaleVersusFemale',
        'GunPointOldVersusYoung',
        'HouseTwenty',
        'InsectEPGRegularTrain',
        'InsectEPGSmallTrain',
        'MelbournePedestrian',
        'PickupGestureWiimoteZ',
        'PigAirwayPressure',
        'PigArtPressure',
        'PigCVP',
        'PLAID',
        'PowerCons',
        'Rock',
        'SemgHandGenderCh2',
        'SemgHandMovementCh2',
        'SemgHandSubjectCh2',
        'ShakeGestureWiimoteZ',
        'SmoothSubspace',
        'UMD'
    ]:
        mean = np.nanmean(train)
        std = np.nanstd(train)
        train = (train - mean) / std
        test = (test - mean) / std

    if use_fft:
        sps = [np.fft.fft(i) for i in train]
        train_fft = np.array([np.stack([sp.real, sp.imag],0) for sp in sps]).transpose(0,2,1)
        sps = [np.fft.fft(i) for i in test]
        test_fft = np.array([np.stack([sp.real, sp.imag],0) for sp in sps]).transpose(0,2,1)
        return train_fft, train_labels, test_fft, test_labels
    return train[..., np.newaxis], train_labels, test[..., np.newaxis], test_labels


def load_UEA(dataset):
    if f'{dataset}.pkl' in os.listdir(f'datasets/UEApkls/'):
        with open(f'./datasets/UEApkls/{dataset}.pkl', 'rb') as fin:
            (train_X, train_y, test_X, test_y) = pickle.load(fin)
            return train_X, train_y, test_X, test_y
    train_data = loadarff(f'datasets/UEA/{dataset}/{dataset}_TRAIN.arff')[0]
    test_data = loadarff(f'datasets/UEA/{dataset}/{dataset}_TEST.arff')[0]
    
    def extract_data(data):
        res_data = []
        res_labels = []
        for t_data, t_label in data:
            t_data = np.array([ d.tolist() for d in t_data ])
            t_label = t_label.decode("utf-8")
            res_data.append(t_data)
            res_labels.append(t_label)
        return np.array(res_data).swapaxes(1, 2), np.array(res_labels)
    
    train_X, train_y = extract_data(train_data)
    test_X, test_y = extract_data(test_data)
    
    scaler = StandardScaler()
    scaler.fit(train_X.reshape(-1, train_X.shape[-1]))
    train_X = scaler.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    test_X = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)
    
    labels = np.unique(train_y)
    transform = { k : i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y)


    with open(f'./datasets/UEApkls/{dataset}.pkl','wb') as fin:
        pickle.dump((train_X, train_y, test_X, test_y),fin)

    return train_X, train_y, test_X, test_y
    
def load_forecast_npy(name, univar=False):
    data = np.load(f'datasets/{name}.npy')    
    if univar:
        data = data[: -1:]

    train_slice = slice(None, int(0.6 * len(data)))
    valid_slice = slice(int(0.6 * len(data)), int(0.8 * len(data)))
    test_slice = slice(int(0.8 * len(data)), None)
    
    scaler = StandardScaler().fit(data[train_slice])
    data = scaler.transform(data)
    data = np.expand_dims(data, 0)

    pred_lens = [24, 48, 96, 288, 672]
    return data, train_slice, valid_slice, test_slice, scaler, pred_lens, 0

def _get_time_features(dt):
    return np.stack([
        dt.minute.to_numpy(),
        dt.hour.to_numpy(),
        dt.dayofweek.to_numpy(),
        dt.day.to_numpy(),
        dt.dayofyear.to_numpy(),
        dt.month.to_numpy(),
        dt.weekofyear.to_numpy(),
    ], axis=1).astype(np.float32)

def load_forecast_csv_lora(univar=False,path =r'datasets/lora/lora-all.csv' ):

    target_feature = [
        'gw1-lp-RSSI2',
        'gw1-lp-SNR2',
        'gw2-lp-RSSI1',
        'gw2-lp-SNR1',
        'gw1-lp-BER2',
        'gw1-lp-PRR2',
        'gw2-lp-BER1',
        'gw2-lp-PRR1']

    data = pd.read_csv(path, index_col='time', parse_dates=True)
    dt_embed = _get_time_features(data.index)
    if univar == True :
        forecast_cols = target_feature[:1]
    else:
        forecast_cols = target_feature

    feat = [data[[col]].to_numpy() for col in forecast_cols]
    feat = np.stack(feat, axis=1).astype(np.float32).reshape([-1,len(forecast_cols)])

    data = feat

    train_slice = slice(None, int(0.6 * len(data)))
    valid_slice = slice(int(0.6 * len(data)), int(0.8 * len(data)))
    test_slice = slice(int(0.8 * len(data)), None)
    
    scaler = StandardScaler().fit(data[train_slice])
    data = scaler.transform(data)

    n_covariate_cols = dt_embed.shape[-1]
    if n_covariate_cols > 0:
        scaler = StandardScaler().fit(dt_embed)
        dt_embed = scaler.transform(dt_embed)

    all_dt_embed = np.expand_dims(dt_embed,axis=0)
    all_data_feat = np.expand_dims(data, 0)
    data = np.concatenate([np.repeat(all_dt_embed, all_data_feat.shape[0], axis=0), all_data_feat], axis=-1)

    pred_lens = [24, 48, 168, 336, 720]

    return data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols


def load_forecast_csv(name, univar=False):
    # data_dir = f'~/data/AutoTCL/datasets/forecast/{name}.csv'
    data = pd.read_csv(data_dir, index_col='date', parse_dates=True)
    data = pd.read_csv(f'datasets/forecast/{name}.csv', index_col='date', parse_dates=True)
    dt_embed = _get_time_features(data.index)
    n_covariate_cols = dt_embed.shape[-1]
    
    if univar:
        if name in ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'):
            data = data[['OT']]
        elif name == 'electricity':
            data = data[['MT_001']]
        elif name =="WTH":
            data = data[['WetBulbCelsius']]
        else:
            data = data.iloc[:, -1:]

    data = data.to_numpy()
    if name == 'ETTh1' or name == 'ETTh2':
        train_slice = slice(None, 12*30*24)
        valid_slice = slice(12*30*24, 16*30*24)
        test_slice = slice(16*30*24, 20*30*24)
    elif name == 'ETTm1' or name == 'ETTm2':
        train_slice = slice(None, 12*30*24*4)
        valid_slice = slice(12*30*24*4, 16*30*24*4)
        test_slice = slice(16*30*24*4, 20*30*24*4)
    else:
        train_slice = slice(None, int(0.6 * len(data)))
        valid_slice = slice(int(0.6 * len(data)), int(0.8 * len(data)))
        test_slice = slice(int(0.8 * len(data)), None)
    
    scaler = StandardScaler().fit(data[train_slice])
    data = scaler.transform(data)
    if name in ('electricity'):
        data = np.expand_dims(data.T, -1)  # Each variable is an instance rather than a feature
    else:
        data = np.expand_dims(data, 0)
    
    if n_covariate_cols > 0:
        dt_scaler = StandardScaler().fit(dt_embed[train_slice])
        dt_embed = np.expand_dims(dt_scaler.transform(dt_embed), 0)
        data = np.concatenate([np.repeat(dt_embed, data.shape[0], axis=0), data], axis=-1)

    if name in ('ETTh1', 'ETTh2', 'WTH'):
        pred_lens = [24, 48, 168, 336, 720]
    elif name in ('electricity'):
        pred_lens = [24, 48, 168, 336]
    else:
        pred_lens = [24, 48, 96, 288, 672]
        
    return data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols

if __name__=="__main__":
    load_forecast_csv_lora(path =r'../datasets/lora/lora-all.csv' )
