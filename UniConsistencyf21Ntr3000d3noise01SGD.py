import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import scipy
import numpy as np
import scipy.io as io
import time
import torch.utils.data as Data
from dltools.mfcnet import FCNet
from dltools.train_eval import FCNetRegressTrain
from dltools.config_common import opt
from dltools.data_generator import Simf1TrainDataArray, Simf1TestDataArray, seed_torch


torch.set_default_tensor_type(torch.DoubleTensor)


Ntr = 3000
Nte = 1000
d = 3
c = 1
ExNum = 50
noise = 0.1

L_seq = np.arange(3, 16, 1)
epoch_seq = [2000] * 13

epoch_max = max(epoch_seq)
n_neuron = d + 1

opt.act_type = 'relu'
opt.init_type = 'kaiming_normal'
opt.batch_size = 50
opt.lr_decay = 1
opt.lr = 0.01
opt.epoch_freq = 100
opt.print_freq = 10000
opt.weight_decay_flag = True
opt.weight_decay = 0.001
opt.criterion = torch.nn.MSELoss()

dir_name = os.path.join(os.path.dirname(__file__), 'synresults')
if not os.path.exists(dir_name):
    os.mkdir(dir_name)

save_file = 'Ntr{}d{}NEpoMax{}noise{}{}_wd{}SGD.mat'.format(Ntr, d, epoch_max, str(noise).replace('.',''),
                                                      opt.init_type, str(opt.weight_decay).replace('.',''))

save_path = os.path.join(dir_name, save_file)

if __name__ == '__main__':
    data_tr_all = np.zeros((Ntr, d+1, ExNum))
    data_te_all = np.zeros((Nte, d+1, ExNum))
    train_rmse = np.zeros((epoch_max+1, len(L_seq), ExNum))
    test_rmse = np.zeros((epoch_max+1, len(L_seq), ExNum))
    for seed in range(ExNum):
        # Generate samples
        seed_torch(seed)
        x_tr, y_tr = Simf1TrainDataArray(Ntr, d, c, noise)
        data_tr_all[:, :, seed] = np.concatenate((x_tr, y_tr), axis=1)
        data_tr = torch.from_numpy(data_tr_all[:, :, seed])
        x_tr = torch.from_numpy(x_tr)
        y_tr = torch.from_numpy(y_tr)
        x_te, y_te = Simf1TestDataArray(Nte, d, c)
        data_te_all[:, :, seed] = np.concatenate((x_te, y_te), axis=1)
        data_te = torch.from_numpy(data_te_all[:, :, seed])

        tr_torch_dataset = Data.TensorDataset(x_tr, y_tr)
        dataTrLoader = Data.DataLoader(tr_torch_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)

        for k in range(len(L_seq)):
            time_start = time.process_time()
            opt.max_epoch = epoch_seq[k]
            L = L_seq[k]
            n_hidden = [n_neuron for _ in range(L-2)]

            model = FCNet(input_dim=d, output_dim=1, n_hiddens=n_hidden, init_type=opt.init_type, act_type=opt.act_type)

            if opt.weight_decay_flag:
                params = [
                    {'params': (p for name, p in model.named_parameters() if 'bias' not in name)},
                    {'params': (p for name, p in model.named_parameters() if 'bias' in name), 'weight_decay': 0.}
                ]
                optimizer = torch.optim.SGD(params=params, lr=opt.lr, weight_decay=opt.weight_decay)
            else:
                optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr)

            train_mse, test_mse = FCNetRegressTrain(model=model, optimizer=optimizer, train_loader=dataTrLoader,
                                      train_data=data_tr, test_data=data_te)

            time_end = time.process_time()
            time_sum = time_end - time_start

            print("Optimizer=SGD\tEx={}\tL={}\tn_neuron={}\tRMSETeMin={}\ttime_cost={}".
                  format(seed, L, n_neuron, min(np.sqrt(test_mse)), time_sum)
                  )

            train_rmse[:(opt.max_epoch+1), k, seed] = np.sqrt(train_mse)
            test_rmse[:(opt.max_epoch+1), k, seed] = np.sqrt(test_mse)

        scipy.io.savemat(save_path,
                         {'train_rmse': train_rmse, 'test_rmse': test_rmse, 'd': d, 'c': c, 'epoch_seq': epoch_seq,
                          'ExNum': ExNum, 'L_seq': L_seq, 'n_neuron': n_neuron, 'noise': noise, 'Nte': Nte,
                          'Ntr': Ntr, 'opt': opt, 'data_tr_all': data_tr_all, 'data_te_all': data_te_all})

    train_rmse_mean = np.mean(train_rmse, axis=2)
    test_rmse_mean = np.mean(test_rmse, axis=2)



    scipy.io.savemat(save_path, {'train_rmse': train_rmse, 'test_rmse':test_rmse, 'd': d, 'c': c, 'epoch_seq': epoch_seq,
                    'ExNum': ExNum, 'L_seq': L_seq, 'n_neuron': n_neuron, 'noise': noise, 'Nte': Nte,
                    'Ntr': Ntr, 'opt': opt, 'train_rmse_mean': train_rmse_mean, 'test_rmse_mean': test_rmse_mean,
                    'data_tr_all': data_tr_all, 'data_te_all': data_te_all})



