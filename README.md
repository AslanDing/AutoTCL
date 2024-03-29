# This code is for the work of [AutoTCL](https://github.com/AI4TS/AI4TS.github.io/blob/main/CameraReadys%201-22%202/19%5CAI4TSAutoTCL%20(3).pdf) and [Parametric Augmentation for Time Series Contrastive Learning(ICLR2024)](https://openreview.net/forum?id=EIPLdFy3vp)
<p align="center">
<img src="./figures/AutoTCL_framework.PNG" width = "700" alt="" align=center />
<br><br>
<b>Figure 1.</b> Overall framework of our work.
</p>

In this paper, we propose [a contrastive learning framework]() with parametric augmentation, AutoTCL, 
which can be adaptively employed to support time series representation learning. 
The proposed approach is [encoder-agnostic](), allowing it to be seamlessly integrated with different backbone
encoders.


## Requirements
We use python 3.9, the packages main contain numpy, scikit-learn, torch, tqdm, matplotlib, pandas. We also provide a 
requirements.txt(contains some other irrelevant packages).

## dataset
1. ETT dataset [(link)](https://github.com/zhouhaoyi/ETDataset)
2. Electricity dataset [(link)](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014), preprocessing by this file [(link)](https://github.com/zhihanyue/ts2vec/blob/main/datasets/preprocess_electricity.py)
3. Weather Dataset [(link)](https://drive.google.com/drive/folders/1ohGYWWohJlOlb2gsGTeEq3Wii2egnEPR), from Informer [(link)](https://github.com/zhouhaoyi/Informer2020)
4. For public datasets, you can also find sources in [TS2Vec](https://github.com/yuezhihan/ts2vec), [CoST](https://github.com/salesforce/CoST).
5. For lora datasets, you can contact with [Professor Mo Sha](https://users.cs.fiu.edu/~msha/).

## usage
```commandline
python train_forecasting_autotcl_cost.py 
--dataset <dataset> 
--gpu <num> 
--seed <random_seed> 
--archive <forecast_csv_univar/forecast_csv>
--load_default <True/False> 

--batch-size <N> 
--lr <encoder_learning_rate> 
--meta_lr <augument_learning_rate>
--mask_mode <noise_style_encoder> 
--augmask_mode <noise_style_augument> 

--repr_dims <embedding_dim>
--hidden_dims <hid_dim> 
--aug_dim <aug_hid_dim> 
--aug_depth <augument_layers> 

--max_train_length <input_time_length> 
--epochs <training_epoch> 
--gumbel_bias <gum_bias>
--gamma_zeta <zeta_in_gumbel>
--ratio_step <ratio_update_between_encoder_augument> 
--hard_mask <hard_gumbel>

--local_weight <local_infonce_encoder>
--reg_weight <reg_weight_hx>
--regular_weight <time_constraint_weight_hx>
```
| args             | explanation                                                                                                       | type   | options                                          | 
|------------------|-------------------------------------------------------------------------------------------------------------------|--------|--------------------------------------------------|
| dataset          | dataset name                                                                                                      | string | ETTh1/ETTh2/ETTm1/electricity/WTH                |
| gpu              | the number of GPU                                                                                                 | int    | -                                                |
| seed             | random seed                                                                                                       | int    | -                                                |
| archive          | univar forecasting or multivar                                                                                    | string | forecast_csv_univar/forecast_csv                 |
| load_default     | load pre-defined hyperparemeters                                                                                  | bool   | True/False                                       |
| batch-size       | the batch size of input                                                                                           | int    | -                                                |
| lr               | learning rate for optimize the encoder network                                                                    | float  | -                                                |
| meta_lr          | learning rate for optimize the augumentation network                                                              | float  | -                                                | 
| mask_mode        | how to add noise for optimize the encoder network                                                                 | string | continuous/mask_last/all_true/all_false/binomial |
| augmask_mode     | how to add noise for optimize the augumentation network                                                           | string | continuous/mask_last/all_true/all_false/binomial |
| repr_dims        | embedding dimension of the encoder network                                                                        | int    | -                                                |
| hidden_dims      | hidden dimension of the encoder network                                                                           | int    | -                                                |
| aug_dim          | hidden dimension of the augument network                                                                          | int    | -                                                |-                                                |
| aug_depth        | NN layers of the augment network                                                                                  | int    | -                                                |
| max_train_length | input time length                                                                                                 | int    | -                                                |
| epochs           | training epochs                                                                                                   | int    | -                                                |
| gumbel_bias      | bias for gumble softmax                                                                                           | float  | -                                                |
| gumbel_zeta      | zeta for gumble softmax                                                                                           | float  | -                                                |
| ratio_step      | the ratio of encoder optimize : augument network optimize                                                         | int    | -                                                |
| hard_mask      | h(x) is hard(binary) or soft                                                                                      | bool    | True/False                                       |
| local_weight      | weight of local infoNCE loss introduced by [InfoTS](https://ojs.aaai.org/index.php/AAAI/article/view/25575/25347) | float    | -                                                |
| reg_weight      | weight of the regularization of h(x)                                                                              | float    | -                                                |
| regular_weight      | weight of the time constraint of h(x)                                                                             | float    | -                                                |



## Main Results
Our experiments conducted on five public datasets and one private dataset. Experiments on univariate forecasting tasks demonstrate the highly
competitive results of our method, with an average 6.5% reduction in MSE and 4.7% in MAE over the leading baselines. In classification tasks, AutoTCL achieves
a 1.2% increase in average accuracy.

In this repo we provide code and default parameters to reproduce our experiments(based on tabl 1 and 2).

<p align="center">
<img src="./figures/AutoTCL_Results.PNG" width = "700" alt="" align=center />
<br><br>
<b>Figure 2.</b> Overall experiments average results.
</p>


<p align="center">
<img src="./figures/AutoTCL_Results1.PNG" width = "700" alt="" align=center />
<br><br>
<b>Figure 3.</b> Detail experiments results in univariate forecasting setting.
</p>

<p align="center">
<img src="./figures/AutoTCL_Results2.PNG" width = "700" alt="" align=center />
<br><br>
<b>Figure 4.</b> Detail experiments results in multivariate forecasting setting.
</p>

## Acknowledgement
Our code are based on the following repositories. We thank the original authors for open-sourcing their work.
1. TS2Vec(https://github.com/zhihanyue/ts2vec)
2. CoST(https://github.com/salesforce/CoST)
3. infoTS(https://github.com/chengw07/InfoTS)


## Citation

If this repo is useful for you, please consider citing our paper as follows:

```
@article{zheng2024parametric,
  title={Parametric Augmentation for Time Series Contrastive Learning},
  author={Zheng, Xu and Wang, Tianchun and Cheng, Wei and Ma, Aitian and Chen, Haifeng and Sha, Mo and Luo, Dongsheng},
  journal={arXiv preprint arXiv:2402.10434},
  year={2024}
}
```

