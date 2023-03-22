import os
import platform
import numpy as np
import torch

import pickle


class Dataloader:
    def __init__(self, pack_file, normalize=False, window_size=60, stride=10, test_set_only=False, train_set_ratio=0.6):
        data = pickle.load(open(pack_file, 'rb'))
        self.test_data_size = data['test'].shape[0]
        self.train_data_size = data['train'].shape[0]
        self.train_data = data['train'].transpose()
        self.test_data = data['test'].transpose()
        self.label_data = data['label']
        train_non_constant_var = []
        test_non_constant_var = []
        for i in self.train_data:
            train_non_constant_var.append(np.unique(i).shape[0])
        for i in self.test_data:
            test_non_constant_var.append(np.unique(i).shape[0])
        self.train_constant_var = np.where(np.array(train_non_constant_var) == 1)
        self.test_constant_var = np.where(np.array(test_non_constant_var) == 1)

        self.train_non_constant_var = np.setdiff1d(np.arange(len(train_non_constant_var)), self.train_constant_var)
        self.test_non_constant_var = np.setdiff1d(np.arange(len(train_non_constant_var)), self.test_constant_var)

        self.nc_train_data = self.train_data[self.train_non_constant_var]
        self.nc_test_data = self.test_data[self.train_non_constant_var]

        self.sensor_num = self.nc_train_data.shape[0]

        self.train_data_std = np.std(self.nc_train_data, axis=1).reshape(-1, 1)
        self.train_data_mean = np.mean(self.nc_train_data, axis=1).reshape(-1, 1)
        self.test_data_std = np.std(self.nc_train_data, axis=1).reshape(-1, 1)
        self.test_data_mean = np.mean(self.nc_train_data, axis=1).reshape(-1, 1)
        if normalize:
            self.nc_train_data = (self.nc_train_data - self.train_data_mean) / self.train_data_std
            self.nc_test_data = (self.nc_test_data - self.test_data_mean) / self.test_data_std

        self.nc_train_data = self.nc_train_data.transpose()
        self.nc_test_data = self.nc_test_data.transpose()
        # print('dataset shape', self.nc_train_data.shape, self.nc_test_data.shape, self.sensor_num)

        if test_set_only:
            sample_set_indices = np.arange(0, self.test_data_size - window_size, stride) \
                .repeat(window_size).reshape(-1, window_size)
            total_samples = sample_set_indices.shape[0]
            window = np.arange(window_size)
            window_mask = np.tile(window, total_samples).reshape(total_samples, -1)
            sample_set_mask = sample_set_indices + window_mask
            label_mask = sample_set_mask[:, -1]
            sample_set = self.nc_test_data[sample_set_mask]
            self.train_set_size = int(sample_set.shape[0] * train_set_ratio)
            self.test_set_size = int(sample_set.shape[0] - self.train_set_size)
            self.train_set = np.expand_dims(sample_set[:self.train_set_size].transpose((0, 2, 1)), axis=-1)
            self.test_set = np.expand_dims(sample_set[self.train_set_size:].transpose((0, 2, 1)), axis=-1)
            self.test_set_label = self.label_data[label_mask][self.train_set_size:]
            # print(self.train_set.shape, self.test_set.shape, self.test_set_label.shape)
        else:
            train_set_indices = np.arange(0, self.train_data_size - window_size, stride) \
                .repeat(window_size).reshape(-1, window_size)
            total_train_samples = train_set_indices.shape[0]
            test_set_indices = np.arange(0, self.test_data_size - window_size, stride) \
                .repeat(window_size).reshape(-1, window_size)
            total_test_samples = test_set_indices.shape[0]
            window = np.arange(window_size)
            train_window_mask = np.tile(window, total_train_samples).reshape(total_train_samples, -1)
            train_set_mask = train_window_mask + window
            test_window_mask = np.tile(window, total_test_samples).reshape(total_test_samples, -1)
            test_set_mask = test_window_mask + window
            label_mask = test_set_mask[:, -1]
            self.train_set = np.expand_dims(self.nc_train_data[train_set_mask].transpose((0, 2, 1)), axis=-1)
            self.test_set = np.expand_dims(self.nc_test_data[test_set_mask].transpose((0, 2, 1)), axis=-1)
            self.test_set_label = self.label_data[label_mask]
            # print(self.train_set.shape, self.test_set.shape, self.test_set_label.shape)

    def load_train_set(self):
        return torch.Tensor(self.train_set)

    def load_test_set(self):
        return torch.Tensor(self.test_set)

    def load_labels(self):
        return torch.Tensor(self.test_set_label)

    def load_n_sensors(self):
        return self.sensor_num


if __name__ == '__main__':
    # a=np.arange(100)
    # b=np.arange(0,90,dtype=int)
    # c=np.arange(10,100,dtype=int)
    # print(b)
    # print(c)
    # print(a[b:c])
    # exit()
    if platform.system() == 'Windows':
        data_dir = 'E:\\Pycharm Projects\\causal.dataset\\data'
        map_dir = 'E:\\Pycharm Projects\\causal.dataset\\maps\\npmap'
    else:
        data_dir = '/remote-home/liuwenbo/pycproj/tsdata/data'
        map_dir = '/remote-home/liuwenbo/pycproj/tsdata/maps/npmap'
    pack_file = os.path.join(data_dir, 'swat', 'raw.data.pkl')
    dataloader = Dataloader(pack_file, test_set_only=False)
    # dataloader.prepare_data(test_set_only=True)
    train_set=dataloader.load_train_set()
    test_set=dataloader.load_test_set()
    labels=dataloader.load_labels()
    print(train_set.shape,test_set.shape,labels.shape)
