import torch
torch.autograd.set_detect_anomaly(True)
from torch.utils.data import TensorDataset, DataLoader
from models.CoTs_encoder import CoSTEncoder as TSEncoder
from scipy.special import softmax
from models.losses import *
from sklearn.metrics import log_loss
import tasks
from models.basicaug import *
LAEGE_NUM = 1e7
import nni
from einops import rearrange, repeat, reduce

class AutoTCL:
    '''The AutoTCL model'''
    
    def __init__(
        self,
        input_dims,
        output_dims=320,
        hidden_dims=64,
        aug_depth=3,
        device='cuda',
        lr=0.001,
        meta_lr = 0.01,
        aug_dim = 16,
        batch_size=16,
        max_train_length=None,
        augmask_mode = 'binomial',
        eval_every_epoch = 20,
        eval_start_epoch = 20,
        aug_net_training = 'PRI',
        gamma_zeta = 0.05,
        hard_mask = True,
        gumbel_bias = 0.001,
        kernels= [1, 2, 4, 8, 16, 32, 64, 128],
        agu_channel = 1
    ):
        ''' Initialize a TS2Vec model.
        
        Args:
            input_dims (int): The input dimension. For a univariate time series, this should be set to 1.
            output_dims (int): The representation dimension.
            hidden_dims (int): The hidden dimension of the encoder.
            depth (int): The number of hidden residual blocks in the encoder.
            device (int): The gpu used for training and inference.
            lr (int): The learning rate.
            meta_lr (int): The learning rate for meta learner.
            batch_size (int): The batch size.
            max_train_length (Union[int, NoneType]): The maximum allowed sequence length for training. For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length>.
        '''
        
        super().__init__()
        self.reg_thres = 0.4
        self.device = device
        self.gumbel_bias = gumbel_bias
        self.lr = lr
        self.batch_size = batch_size
        self.max_train_length = max_train_length
        self._net = TSEncoder(input_dims=input_dims, output_dims=output_dims,kernels=kernels,length=max_train_length).to(self.device)
        
        
        self.augsharenet = TSEncoder(input_dims=input_dims, output_dims=aug_dim,kernels=kernels,length=max_train_length,
                        hidden_dims=hidden_dims, depth=aug_depth,mask_mode=augmask_mode).to(self.device)

        self.factor_augnet = torch.nn.Sequential(torch.nn.Linear(aug_dim,agu_channel),torch.nn.Sigmoid()).to(self.device)########New  h(x)

        self.augmentation_projector = torch.nn.Sequential(torch.nn.Linear(aug_dim,agu_channel),torch.nn.Sigmoid()).to(self.device) ########New g(x)


        self.net = torch.optim.swa_utils.AveragedModel(self._net)
        self.net.update_parameters(self._net)

        self.aug_net_training = aug_net_training
        self.hard_mask = hard_mask
        self.n_epochs = 0
        self.n_iters = 0

        self.meta_lr = meta_lr
        self.gamma_zeta = -gamma_zeta
        self.zeta = 1.0

        self.CE = torch.nn.CrossEntropyLoss()
        self.BCE = torch.nn.BCEWithLogitsLoss()
        self.eval_every_epoch = eval_every_epoch
        self.eval_start_epoch = eval_start_epoch

        # self.mmd_loss = MMDLoss()

    def get_dataloader(self,data,shuffle=False, drop_last=False):

        # pre_process to return data loader

        if self.max_train_length is not None:
            sections = data.shape[1] // self.max_train_length
            if sections >= 2:
                data = np.concatenate(split_with_nan(data, sections, axis=1), axis=0)

        temporal_missing = np.isnan(data).all(axis=-1).any(axis=0)
        if temporal_missing[0] or temporal_missing[-1]:
            data = centerize_vary_length_series(data)

        data = data[~np.isnan(data).all(axis=2).all(axis=1)]
        data = np.nan_to_num(data)
        dataset = TensorDataset(torch.from_numpy(data).to(torch.float))
        loader = DataLoader(dataset, batch_size=min(self.batch_size, len(dataset)),shuffle=shuffle, drop_last=drop_last)
        return data, dataset, loader

    def _sample_graph(self, sampling_weights, temperature=1.0, bias=0.0, training=True):
        """
        Implementation of the reparamerization trick to obtain a sample graph while maintaining the posibility to backprop.
        :param sampling_weights: Weights provided by the mlp
        :param temperature: annealing temperature to make the procedure more deterministic
        :param bias: Bias on the weights to make samplign less deterministic
        :param training: If set to false, the samplign will be entirely deterministic
        :return: sample graph
        """
        if training:
            bias = bias + self.gumbel_bias  # If bias is 0, we run into problems
            eps = (bias - (1 - bias)) * torch.rand(sampling_weights.size()).to(sampling_weights.device) + (1 - bias)
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            gate_inputs = (gate_inputs + sampling_weights) / temperature
            graph = torch.sigmoid(gate_inputs)
        else:
            graph = torch.sigmoid(sampling_weights)

        stretched_values = graph * (self.zeta - self.gamma_zeta) + self.gamma_zeta
        cliped = torch.clip(
            stretched_values,
            max=1.0,
            min=0.0)

        return cliped


    def get_features(self, x, training = True, n_epochs=-1, mask=None):

        embedding = self.augsharenet(x)
        weight_h = self.factor_augnet(embedding)
        weight_s = self.augmentation_projector(embedding)

        
        mask_h = self._sample_graph(weight_h,training= training)

        if self.hard_mask:
            hard_mask_h = (torch.sign(mask_h-0.5)+1)/2
            # print(hard_mask_h)
#             mask_h = (mask_h-hard_mask_h).detach()+hard_mask_h
            mask_h = (hard_mask_h - mask_h).detach()+mask_h
        ax = weight_s * mask_h * x  # augmented x'

        
        if torch.isnan(ax).any() or torch.isnan(x).any():
            exit(1)

        # note: I add mask
        out1 = self._net(x,mask)  # representation
        out2 = self._net(ax,mask)  # representation of augmented x'
        return x, ax, out1, out2, weight_h
    def regular_consistency(self,weight):

        B,T,C = weight.shape

        # near
        select0 = torch.randint(1,T-2,[B,])
        left = select0 - 1
        right = select0 + 1
        select1 = torch.randint(1,T-2,[B,])
        #select1 = torch.randint(1,T-1,B)
        mask = torch.where((select1-select0)>1,torch.ones_like(select1),torch.zeros_like(select0)).to(weight.device)

        # near difference
        diff = mask.reshape(1,B,1)*torch.abs(weight[:,select0,:]-weight[:,select1,:]) + \
               torch.abs(weight[:,select0,:]-weight[:,left,:]) + \
               torch.abs(weight[:,select0,:]-weight[:,right,:]) + \
               (1-mask).reshape(1,B,1)*(1-torch.abs(weight[:,select0,:]-weight[:,select1,:]))

        return diff.mean()

    # calculate mutual information MI(v,x)
    def MI(self, data_loader):
        ori_training = self._net.training
        self._net.eval()
        cum_vx = 0
        zvs = []
        zxs = []
        size = 0
        with torch.no_grad():
            for batch in data_loader:
                x = batch[0]
                if self.max_train_length is not None and x.size(1) > self.max_train_length:
                    window_offset = np.random.randint(x.size(1) - self.max_train_length + 1)
                    x = x[:, window_offset: window_offset + self.max_train_length]
                x = x.to(self.device)
                outv, outx = self.get_features(x)
                vx_infonce_loss = L1out(outv, outx) * x.size(0)
                size +=x.size(0)

                zv = F.max_pool1d(outv.transpose(1, 2).contiguous(), kernel_size=outv.size(1)).transpose(1,2).squeeze(1)
                zx = F.max_pool1d(outx.transpose(1, 2).contiguous(), kernel_size=outx.size(1)).transpose(1,2).squeeze(1)

                cum_vx += vx_infonce_loss.item()
                zvs.append(zv.cpu().numpy())
                zxs.append(zx.cpu().numpy())

        MI_vx_loss = cum_vx / size
        zvs = np.concatenate(zvs,0)
        zxs = np.concatenate(zxs,0)

        if ori_training:
            self._net.train()
        return zvs,MI_vx_loss

    def fit(self, train_data, n_epochs=None, n_iters=None,task_type='classification',
            valid_dataset=None, miverbose=None,
            # train_labels = None, split_number=8,
            # meta_epoch=2,meta_beta=1.0,verbose=False,beta=1.0,
            ratio_step=1,lcoal_weight=0.1,reg_weight = 0.001,
            regular_weight = 0.001,evalall =  False):
        '''
        
        Args:
            train_data (numpy.ndarray): The training data. It should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            n_epochs (Union[int, NoneType]): The number of epochs. When this reaches, the training stops.
            n_iters (Union[int, NoneType]): The number of iterations. When this reaches, the training stops. If both n_epochs and n_iters are not specified, a default setting would be used that sets n_iters to 200 for a dataset with size <= 100000, 600 otherwise.
            verbose (bool): Whether to print the training loss after each epoch.
            beta (float): trade-off between global and local contrastive.
            valid_dataset:  (train_data, train_label,test_data,test_label) for Classifier.
            miverbose (bool): Whether to print the information of meta-learner
            meta_epoch (int): meta-parameters are updated every meta_epoch epochs
            meta_beta (float): trade-off between high variety and high fidelity.
            task_type (str): downstream task
        Returns:
            crietira.
        '''

        # check the input formation
        assert train_data.ndim == 3

        do_valid = False if valid_dataset is None else True

        # default param for n_iters
        if n_iters is None and n_epochs is None:
            n_iters = 200 if train_data.size <= 100000 else 600

        train_data,train_dataset,train_loader =  self.get_dataloader(train_data,shuffle=True,drop_last=True)

        if task_type=='classification' and valid_dataset is not None:
            cls_train_data, cls_train_labels, cls_test_data, cls_test_labels = valid_dataset
            cls_train_data,cls_train_dataset,cls_train_loader = self.get_dataloader(cls_train_data,shuffle=False,drop_last=False)

        import itertools
        params = itertools.chain(self.augsharenet.parameters(),self.factor_augnet.parameters(),self.augmentation_projector.parameters())
        meta_optimizer = torch.optim.AdamW(params, lr=self.meta_lr)

        optimizer = torch.optim.AdamW(self._net.parameters(), lr=self.lr)

        self.t0 = 1.0
        self.t1 = 1.0

        acc_log = []
        vy_log = []
        vx_log = []
        loss_log = []

        mses = []
        maes = []

        def eval(final=False,s = True):
            self._net.eval()
            self.factor_augnet.eval()
            # try:
            if task_type == 'classification':
                out, eval_res = tasks.eval_classification(self, cls_train_data, cls_train_labels, cls_test_data,
                                                          cls_test_labels, eval_protocol='svm')
                clf = eval_res['clf']
                zvs, MI_vx_loss = self.MI(cls_train_loader)

                v_pred = softmax(clf.decision_function(zvs), -1)
                MI_vy_loss = log_loss(cls_train_labels, v_pred)
                v_acc = clf.score(zvs, cls_train_labels)

                vx_log.append(MI_vx_loss)
                vy_log.append(MI_vy_loss)

                acc_log.append(eval_res['acc'])

                if miverbose:
                    print('acc %.3f (max)vx %.3f (min)vy %.3f (max)vacc %.3f' % (
                    eval_res['acc'], MI_vx_loss, MI_vy_loss, v_acc))
            elif task_type == 'forecasting':
                if not final:
                    valid_dataset_during_train = valid_dataset[0],valid_dataset[1],valid_dataset[2],valid_dataset[3],valid_dataset[4],[valid_dataset[5][-1]],valid_dataset[6]
                    out, eval_res = tasks.eval_forecasting(self, *valid_dataset_during_train)
                else:
                    if s :
                        out, eval_res = tasks.eval_forecasting(self, *valid_dataset)
                    else:
                        valid_dataset_during_train = valid_dataset[0], valid_dataset[1], valid_dataset[2], \
                                                     valid_dataset[3], valid_dataset[4], [valid_dataset[5][0]], \
                                                     valid_dataset[6]
                        out, eval_res = tasks.eval_forecasting(self, *valid_dataset_during_train)

                res = eval_res['ours']
                mse = sum([res[t]['norm']['MSE'] for t in res]) / len(res)
                mae = sum([res[t]['norm']['MAE'] for t in res]) / len(res)
                mses.append(mse)
                maes.append(mae)
                for key in eval_res['ours']:
                    print(key,eval_res['ours'][key])
                print("avg.", mse, mae)
                print("avg. total", mse + mae)



        while True:
            if n_epochs is not None and self.n_epochs >= n_epochs:
                break

            cum_loss = 0
            n_epoch_iters = 0

            interrupted = False


            for batch in train_loader:
                if n_iters is not None and self.n_iters >= n_iters:
                    interrupted = True
                    break

                x = batch[0]
                if self.max_train_length is not None and x.size(1) > self.max_train_length:
                    window_offset = np.random.randint(x.size(1) - self.max_train_length + 1)
                    x = x[:, window_offset : window_offset + self.max_train_length]
                x = x.to(self.device)

                if self.n_iters % ratio_step == 0 :
                    self._net.eval()
                    self.factor_augnet.train()
                    if self.aug_net_training=='PRI':
                        meta_optimizer.zero_grad()
                        x_,ax_,outx,outv,weight_h = self.get_features(x,mask='all_true')
                        vx_distance = mmdx(outx,outv)
                        regular = self.regular_consistency(weight_h)
                        reg_loss = torch.nn.functional.relu(torch.sum(weight_h,dim=-1).mean()-self.reg_thres)
                        aloss = vx_distance + reg_weight * reg_loss + regular_weight * regular
                        aloss.backward()
                        meta_optimizer.step()
                        # print("PRI aug loss ",vx_distance.item(),torch.sum(weight_h,dim=-1).mean().item())
                    elif self.aug_net_training=='Adversarial':
                        x_,ax_,outx,outv,weight_h = self.get_features(x,mask='all_true')
                        meta_optimizer.zero_grad()
                        vx_distance = -1*infoNCE(outx,outv,temperature=self.t0)
                        reg_loss = torch.sum(weight_h,dim=-1).mean()
                        aloss = vx_distance
                        aloss.backward()
                        meta_optimizer.step()
                        # print("Adversarial aug loss ",vx_distance.item(),reg_loss.item())

                self._net.train()
                self.factor_augnet.eval()

                optimizer.zero_grad()
                x_, ax_, outx, outv, _ = self.get_features(x, n_epochs=n_epochs)
                local_loss = local_infoNCE(outx, outv)
                loss = infoNCE(outx, outv, temperature=self.t1)
                all_loss = loss + lcoal_weight * local_loss
                all_loss.backward()
                optimizer.step()
                # print("agree loss ", loss.item(), local_loss.item())

                self.net.update_parameters(self._net)
                    
                cum_loss += loss.item()
                n_epoch_iters += 1

                self.n_iters += 1

            self.n_epochs += 1
            print("epoch ", self.n_epochs)
            if self.n_epochs%self.eval_every_epoch==0 and self.n_epochs > self.eval_start_epoch:
                # print("epoch ",self.n_epochs)
                if do_valid:
                    eval(evalall)

            if interrupted:
                break

            cum_loss /= n_epoch_iters
            loss_log.append(cum_loss)

        if do_valid:
            eval(True)

        if task_type == 'classification':
            return loss_log,acc_log,vx_log,vy_log
        else:
            return mses,maes

    def encode(self, data, mask=None, batch_size=None):
        ''' Compute representations using the model.

        Args:
            data (numpy.ndarray): This should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            mask (str): The mask used by encoder can be specified with this parameter. This can be set to 'binomial', 'continuous', 'all_true', 'all_false' or 'mask_last'.
            encoding_window (Union[str, int]): When this param is specified, the computed representation would the max pooling over this window. This can be set to 'full_series', 'multiscale' or an integer specifying the pooling kernel size.
            casual (bool): When this param is set to True, the future informations would not be encoded into representation of each timestamp.
            sliding_padding (int): This param specifies the contextual data length used for inference every sliding windows.
            batch_size (Union[int, NoneType]): The batch size used for inference. If not specified, this would be the same batch size as training.
        Returns:
            repr: The representations for data.
        '''


        assert data.ndim == 3
        if batch_size is None:
            batch_size = self.batch_size
        n_samples, ts_l, _ = data.shape

        org_training = self.net.training
        self.net.eval()

        dataset = TensorDataset(torch.from_numpy(data).to(torch.float))
        loader = DataLoader(dataset, batch_size=batch_size)

        with torch.no_grad():
            output = []
            for batch in loader:
                x = batch[0]
                out = self.net(x.to(self.device, non_blocking=True), mask)
                out = F.max_pool1d(out.transpose(1, 2), kernel_size=out.size(1)).transpose(1, 2).cpu()
                out = out.squeeze(1)

                output.append(out)

            output = torch.cat(output, dim=0)

        self.net.train(org_training)
        return output.numpy()

    def casual_encode(self, data, encoding_window=None, mask=None, sliding_length=None, sliding_padding=0,  batch_size=None):
        ''' Compute representations using the model.

        Args:
            data (numpy.ndarray): This should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            mask (str): The mask used by encoder can be specified with this parameter. This can be set to 'binomial', 'continuous', 'all_true', 'all_false' or 'mask_last'.
            encoding_window (Union[str, int]): When this param is specified, the computed representation would the max pooling over this window. This can be set to 'full_series', 'multiscale' or an integer specifying the pooling kernel size.
            casual (bool): When this param is set to True, the future informations would not be encoded into representation of each timestamp.
            sliding_padding (int): This param specifies the contextual data length used for inference every sliding windows.
            batch_size (Union[int, NoneType]): The batch size used for inference. If not specified, this would be the same batch size as training.
        Returns:
            repr: The representations for data.
        '''
        casual = True
        if batch_size is None:
            batch_size = self.batch_size
        n_samples, ts_l, _ = data.shape

        org_training = self.net.training
        self.net.eval()

        dataset = TensorDataset(torch.from_numpy(data).to(torch.float))
        loader = DataLoader(dataset, batch_size=batch_size)

        with torch.no_grad():
            output = []
            for batch in loader:
                x = batch[0]
                if sliding_length is not None:
                    reprs = []
                    if n_samples < batch_size:
                        calc_buffer = []
                        calc_buffer_l = 0
                    for i in range(0, ts_l, sliding_length):
                        l = i - sliding_padding
                        r = i + sliding_length + (sliding_padding if not casual else 0)
                        x_sliding = torch_pad_nan(
                            x[:, max(l, 0): min(r, ts_l)],
                            left=-l if l < 0 else 0,
                            right=r - ts_l if r > ts_l else 0,
                            dim=1
                        )
                        if n_samples < batch_size:
                            if calc_buffer_l + n_samples > batch_size:
                                out = self._eval_with_pooling(
                                    torch.cat(calc_buffer, dim=0),
                                    mask,
                                    slicing=slice(sliding_padding, sliding_padding + sliding_length),
                                    encoding_window=encoding_window
                                )
                                reprs += torch.split(out, n_samples)
                                calc_buffer = []
                                calc_buffer_l = 0
                            calc_buffer.append(x_sliding)
                            calc_buffer_l += n_samples
                        else:
                            out = self._eval_with_pooling(
                                x_sliding,
                                mask,
                                slicing=slice(sliding_padding, sliding_padding + sliding_length),
                                encoding_window=encoding_window
                            )
                            reprs.append(out)

                    if n_samples < batch_size:
                        if calc_buffer_l > 0:
                            out = self._eval_with_pooling(
                                torch.cat(calc_buffer, dim=0),
                                mask,
                                slicing=slice(sliding_padding, sliding_padding + sliding_length),
                                encoding_window=encoding_window
                            )
                            reprs += torch.split(out, n_samples)
                            calc_buffer = []
                            calc_buffer_l = 0

                    out = torch.cat(reprs, dim=1)
                    if encoding_window == 'full_series':
                        out = F.max_pool1d(
                            out.transpose(1, 2).contiguous(),
                            kernel_size=out.size(1),
                        ).squeeze(1)
                else:
                    out = self._eval_with_pooling(x, mask, encoding_window=encoding_window)
                    if encoding_window == 'full_series':
                        out = out.squeeze(1)

                output.append(out)

            output = torch.cat(output, dim=0)

        self.net.train(org_training)
        return output.numpy()

    def save(self, fn):
        ''' Save the model to a file.
        
        Args:
            fn (str): filename.
        '''
        torch.save(self.net.state_dict(), fn)
    
    def load(self, fn):
        ''' Load the model from a file.
        
        Args:
            fn (str): filename.
        '''
        state_dict = torch.load(fn, map_location=self.device)
        self.net.load_state_dict(state_dict)

    def _eval_with_pooling(self, x, mask=None, slicing=None, encoding_window=None):
        out = self.net(x.to(self.device, non_blocking=True), mask)
        if encoding_window == 'full_series':
            if slicing is not None:
                out = out[:, slicing]
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size=out.size(1),
            ).transpose(1, 2)

        elif isinstance(encoding_window, int):
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size=encoding_window,
                stride=1,
                padding=encoding_window // 2
            ).transpose(1, 2)
            if encoding_window % 2 == 0:
                out = out[:, :-1]
            if slicing is not None:
                out = out[:, slicing]

        elif encoding_window == 'multiscale':
            p = 0
            reprs = []
            while (1 << p) + 1 < out.size(1):
                t_out = F.max_pool1d(
                    out.transpose(1, 2),
                    kernel_size=(1 << (p + 1)) + 1,
                    stride=1,
                    padding=1 << p
                ).transpose(1, 2)
                if slicing is not None:
                    t_out = t_out[:, slicing]
                reprs.append(t_out)
                p += 1
            out = torch.cat(reprs, dim=-1)

        else:
            if slicing is not None:
                out = out[:, slicing]

        return out.cpu()