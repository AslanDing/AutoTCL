config_etth2_uni = {'batch_size': 32, 'lr': 1e-06, 'meta_lr': 0.001, 'mask_mode': 'continuous',
                    'augmask_mode': 'mask_last', 'bias_init': 0.9769944119927584, 'local_weight': 1e-05,
                    'reg_weight': 0.003, 'regular_weight': 3e-06, 'dropout': 0.07481866599455227,
                    'augdropout': 0.04299077431318535, 'hidden_dims': 128, 'max_train_length': 512,
                    'depth': 9, 'aug_depth': 3, 'gamma_zeta': 0.022899732364593785, 'aug_dim': 16,
                    'ratio_step': 1, 'gumbel_bias': 0.001, 'hard_mask': False, 'epochs': 800}

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
    if dataset == "ETTh2":
        if univar :
            config_out = merge_parameter(config, config_etth2_uni)
    return config_out

