# these paramer are test under torch.backends.cudnn.deterministic=True,
# however, this setting will cause running slow
# for config_elec, the results are test under deterministic=False

# 
config_etth1_uni = {'batch_size': 32, 'lr': 1e-05, 'meta_lr': 0.012, 'mask_mode': 'mask_last',
                    'augmask_mode': 'mask_last', 'bias_init': 0.90, 'local_weight': 0.3,
                    'reg_weight': 0.003, 'regular_weight': 0, 'dropout': 0.1,
                    'augdropout': 0.1, 'hidden_dims': 64, 'max_train_length': 257,
                    'depth': 10, 'aug_depth': 1, 'gamma_zeta': 0.05, 'aug_dim': 16,'seed':42,
                    'ratio_step': 1, 'gumbel_bias': 0.05, 'hard_mask': True, 'epochs': 100}
# avg. total 1.2429398576236532 avg. 0.6555039071646586 0.5874359504589945
config_etth1 = {'batch_size': 32, 'lr': 1e-05, 'meta_lr': 0.012, 'mask_mode': 'mask_last',
                    'augmask_mode': 'mask_last', 'bias_init': 0.90, 'local_weight': 0.3,
                    'reg_weight': 0.2, 'regular_weight': 0, 'dropout': 0.1,
                    'augdropout': 0.1, 'hidden_dims': 128, 'max_train_length': 257,
                    'depth': 10, 'aug_depth': 1, 'gamma_zeta': 0.05, 'aug_dim': 16,'seed':42,
                    'ratio_step': 1, 'gumbel_bias': 0.05, 'hard_mask': True, 'epochs': 35}



# avg. total 0.465516948684493   avg. 0.16127922624228738 0.30423772244220565
config_etth2_uni = {'batch_size': 32, 'lr': 1e-05, 'meta_lr': 0.01, 'mask_mode': 'continuous',
                    'augmask_mode': 'mask_last', 'bias_init': 0.90, 'local_weight': 0.1,
                    'reg_weight': 0.1, 'regular_weight': 0, 'dropout': 0.3,
                    'augdropout': 0.2, 'hidden_dims': 128, 'max_train_length': 512,
                    'depth': 9, 'aug_depth': 2, 'gamma_zeta': 0.02, 'aug_dim': 16,'seed':42,
                    'ratio_step': 1, 'gumbel_bias': 0.05, 'hard_mask': False, 'epochs': 80}
# avg. total 2.0639568490668982   avg. 1.239253840047174 0.8247030090197244
config_etth2 = {'batch_size': 64, 'lr': 0.00003, 'meta_lr': 0.1, 'mask_mode': 'continuous',
                    'augmask_mode': 'continuous', 'bias_init': 0.90, 'local_weight': 0.000001,
                    'reg_weight': 0.0, 'regular_weight': 0.000001, 'dropout': 0.3,
                    'augdropout': 0.1, 'hidden_dims': 8, 'max_train_length': 256,
                    'depth': 9, 'aug_depth': 5, 'gamma_zeta': 0.0009, 'aug_dim': 8,'seed':42,
                    'ratio_step': 1, 'gumbel_bias': 0.01, 'hard_mask': False, 'epochs': 15}



# avg. total 0.20306313270058415 avg. 0.04760412978270486 0.15545900291787929
config_ettm1_uni = {'batch_size': 32, 'lr': 1e-05, 'meta_lr': 0.01, 'mask_mode': 'continuous',
                    'augmask_mode': 'mask_last', 'bias_init': 0.90, 'local_weight': 0.1,
                    'reg_weight': 0.1, 'regular_weight': 0, 'dropout': 0.2,
                    'augdropout': 0.2, 'hidden_dims': 128, 'max_train_length': 512,
                    'depth': 9, 'aug_depth': 1, 'gamma_zeta': 0.02, 'aug_dim': 16,'seed':42,
                    'ratio_step': 1, 'gumbel_bias': 0.05, 'hard_mask': False, 'epochs': 190}
# avg. total 0.9002543626158892 avg. 0.4405296853946844 0.45972467722120475
config_ettm1 = {'batch_size': 128, 'lr': 0.0001, 'meta_lr': 0.000003, 'mask_mode': 'continuous', 
                    'augmask_mode': 'all_true', 'bias_init': 0.6, 'local_weight': 0.001,
                    'reg_weight': 0.000001, 'regular_weight': 0.00003, 'dropout': 0.3,
                    'augdropout': 0.1, 'hidden_dims': 32, 'max_train_length': 1024,
                    'depth': 10, 'aug_depth': 1, 'gamma_zeta': 0.0006, 'aug_dim': 128,'seed':42,
                    'ratio_step': 2, 'gumbel_bias': 0.2, 'hard_mask': False, 'epochs': 420}

# avg. total 0.7152480190155328 avg. 0.3687407123505206 0.3465073066650122
config_elec_uni = {'batch_size': 64, 'lr': 1e-05, 'meta_lr': 0.01, 'mask_mode': 'continuous',
                    'augmask_mode': 'binomial', 'bias_init': 0.90, 'local_weight': 0.1,
                    'reg_weight': 0.1, 'regular_weight': 0, 'dropout': 0.3,
                    'augdropout': 0.2, 'hidden_dims': 128, 'max_train_length': 257,
                    'depth': 9, 'aug_depth': 2, 'gamma_zeta': 0.02, 'aug_dim': 16,'seed':42,
                    'ratio_step': 1, 'gumbel_bias': 0.05, 'hard_mask': False, 'epochs': 110}
# todo: to be updated
config_elec = {'batch_size': 32, 'lr': 1e-05, 'meta_lr': 0.01, 'mask_mode': 'continuous',
                    'augmask_mode': 'mask_last', 'bias_init': 0.90, 'local_weight': 0.1,
                    'reg_weight': 0.1, 'regular_weight': 0, 'dropout': 0.3,
                    'augdropout': 0.2, 'hidden_dims': 128, 'max_train_length': 257,
                    'depth': 9, 'aug_depth': 2, 'gamma_zeta': 0.02, 'aug_dim': 16,'seed':42,
                    'ratio_step': 1, 'gumbel_bias': 0.05, 'hard_mask': False, 'epochs': 200}


# avg. total 0.4474360587106554  avg.  0.15990516940715715 0.28753088930349824
config_WTH_uni = {'batch_size': 256, 'lr': 3e-03, 'meta_lr': 0.01, 'mask_mode': 'all_True',
                    'augmask_mode': 'binomial', 'bias_init': 0.1, 'local_weight': 0.0003,
                    'reg_weight': 0.03, 'regular_weight': 0.0003, 'dropout': 0.1,
                    'augdropout': 0.1, 'hidden_dims': 128, 'max_train_length': 256,
                    'depth': 8, 'aug_depth': 1, 'gamma_zeta': 0.499, 'aug_dim': 8,'seed':42,
                    'ratio_step': 1, 'gumbel_bias': 0.0006, 'hard_mask': True, 'epochs': 20}
# avg. total 0.8784346657087241  avg.  0.4220096324938378 0.45642503321488626
config_WTH = {'batch_size': 64, 'lr': 0.003, 'meta_lr': 0.000001, 'mask_mode': 'binomial',
                    'augmask_mode': 'binomial', 'bias_init': 0.3, 'local_weight': 0.1,
                    'reg_weight': 0.0003, 'regular_weight': 0.3, 'dropout': 0.3,
                    'augdropout': 0.2, 'hidden_dims': 16, 'max_train_length': 512,
                    'depth': 6, 'aug_depth': 2, 'gamma_zeta': 0.0009, 'aug_dim': 256,'seed':42,
                    'ratio_step': 32, 'gumbel_bias': 0.001, 'hard_mask': False, 'epochs': 20}



def merge_parameter(base_params, override_params):
    """
    Update the parameters in ``base_params`` with ``override_params``.
    Can be useful to override parsed command line arguments.

    Parameters
    ----------
    base_params : namespace or dict
        Base parameters. A key-value mapping.
    override_params : dict or None
        Parameters to override. Usually the parameters got from ``get_next_parameters()``.
        When it is none, nothing will happen.

    Returns
    -------
    namespace or dict
        The updated ``base_params``. Note that ``base_params`` will be updated inplace. The return value is
        only for convenience.
    """
    if override_params is None:
        return base_params
    is_dict = isinstance(base_params, dict)
    for k, v in override_params.items():
        if is_dict:
            if k not in base_params:
                raise ValueError('Key \'%s\' not found in base parameters.' % k)
            v = _ensure_compatible_type(k, base_params[k], v)
            base_params[k] = v
        else:
            if not hasattr(base_params, k):
                raise ValueError('Key \'%s\' not found in base parameters.' % k)
            v = _ensure_compatible_type(k, getattr(base_params, k), v)
            setattr(base_params, k, v)
    return base_params

def _ensure_compatible_type(key, base, override):
    if base is None:
        return override
    if isinstance(override, type(base)):
        return override
    if isinstance(base, float) and isinstance(override, int):
        return float(override)
    base_type = type(base).__name__
    override_type = type(override).__name__
    raise ValueError(f'Expected "{key}" in override parameters to have type {base_type}, but found {override_type}')

def merege_config(config, dataset, univar = True):
    config_out = config
    if dataset == "ETTh1":
        if univar :
            config_out = merge_parameter(config, config_etth1_uni)
        else:
            config_out = merge_parameter(config, config_etth1)
    elif dataset == "ETTh2":
        if univar :
            config_out = merge_parameter(config, config_etth2_uni)
        else:
            config_out = merge_parameter(config, config_etth2)
    elif dataset == "ETTm1":
        if univar :
            config_out = merge_parameter(config, config_ettm1_uni)
        else:
            config_out = merge_parameter(config, config_ettm1)
    elif dataset == "electricity":
        if univar :
            config_out = merge_parameter(config, config_elec_uni)
        else:
            config_out = merge_parameter(config, config_elec)
    elif dataset == "WTH":
        if univar :
            config_out = merge_parameter(config, config_WTH_uni)
        else:
            config_out = merge_parameter(config, config_WTH)
    else:
        print("not pre-config for current dataset")

    return config_out

