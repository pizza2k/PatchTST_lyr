import os
import numpy as np
import pandas as pd
import math
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, fake=None, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 test_year=0, calculate_MSE=0):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 512
            self.label_len = 192
            self.pred_len = 192
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        self.test_year = test_year
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]   
            
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.calculate_MSE = calculate_MSE
        self.df_raw_original = None
        self.df_raw_cols = None
        self.test_start = 0
        
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        
        if self.set_type == 2:
            self.df_raw_original = df_raw
            self.df_raw_cols = df_raw.columns.tolist()
            
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)

        idx1 = idx2 = idx3 = 0
        
        if self.test_year == 0:
            num_train = int(len(df_raw) * 0.7)
            num_test = int(len(df_raw) * 0.2)
            num_vali = len(df_raw) - num_train - num_test
            idx1 = num_train
            idx2 = num_train + num_vali
            idx3 = len(df_raw)
            print(f"   数据总长度: {len(df_raw)}")
            print(f"   num_train: {num_train}")
            print(f"   num_test: {num_test}")
            print(f"   num_vali: {num_vali}")
            
        else:
            df_raw['date'] = pd.to_datetime(df_raw['date'])
            df_raw['year'] = df_raw['date'].dt.year
            df_raw['month'] = df_raw['date'].dt.month
            df_raw['day'] = df_raw['date'].dt.day
            df_raw['hour'] = df_raw['date'].dt.hour
            test_idx = np.where(df_raw['year'] == self.test_year)[0]
            idx2 = test_idx[0]
            # idx3 = math.ceil((test_idx[-1] + 1 - self.seq_len) / self.pred_len) * self.pred_len + self.seq_len
            # idx3 = math.ceil((test_idx[-1] - test_idx[0] + 1) / self.pred_len) *  self.pred_len + test_idx[0]
            idx3 = test_idx[-1] + 1
            idx1 = int(idx2 * 7/9)
            
        border1s = [0, idx1 - self.seq_len, idx2 - self.seq_len]
        border2s = [idx1, idx2, idx3]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        self.test_start = idx2 - self.seq_len
        
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            # print(self.scaler.mean_)
            # exit()
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        self.test_dates = df_raw['date'].values[border1:border2] # 保存测试样例的日期

    def __getitem__(self, index):
        if self.calculate_MSE:
            s_begin = index * self.pred_len
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len
            
            if r_end > len(self.data_x):
                # 调整到最后一个可能的位置
                r_end = len(self.data_x)
                s_end = r_end - self.pred_len
                s_begin = s_end - self.seq_len
                r_begin = s_end - self.label_len
    
            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_y[r_begin:r_end]
            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]

            return seq_x, seq_y, seq_x_mark, seq_y_mark
        else:
            s_begin = index
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len
    
            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_y[r_begin:r_end]
            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]

            return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        if self.calculate_MSE:
            return math.ceil((len(self.data_x) - self.seq_len)/self.pred_len)
        else:
            return len(self.data_x) - self.seq_len - self.pred_len + 1 
            
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def get_df_raw(self, index):
        if self.calculate_MSE:
            begin = index * self.pred_len + self.test_start
            end = begin + self.seq_len + self.pred_len
            
            if end > len(self.df_raw_original):
                end = len(self.df_raw_original)
                begin = end - (self.seq_len + self.pred_len)
            
            return self.df_raw_original.iloc[begin:end].copy()

        else:
            begin = index + self.test_start
            end = begin + self.seq_len + self.pred_len
            
            return self.df_raw_original.iloc[begin:end].copy()
        
    def output_tail_by_id(self, item_id_array, folder_path = "tail_sample"):
        if not hasattr(self, 'df_raw_original'):
            print("错误：原始数据未加载")
            return None
            
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        samples_list = []
        for sample_id in item_id_array:
            sample_df = self.get_df_raw(sample_id)
            if sample_df is not None and not sample_df.empty:
                samples_list.append(sample_df.copy())
     
        if samples_list:
            tail_samples = pd.concat(samples_list, ignore_index=True)
            output_path = os.path.join(folder_path, f"tail_samples_{self.test_year}.csv")
            tail_samples.to_csv(output_path, index=False)
            print(f"已保存 {len(item_id_array)} 个长尾样本到 {output_path}")
            return tail_samples
            
        else:
            print("没有找到长尾样本")
            return None

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Custom_Test(Dataset):
    def __init__(self, root_path, fake=None, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', test_year=0, calculate_MSE=0):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 512
            self.label_len = 192
            self.pred_len = 192
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.fake = fake
        if fake is not None:
            self.fake_root_path, self.fake_data_path, self.fake_weight = fake

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_fake_data__(self):
        """读取并处理fake数据"""
        fake_df = pd.read_csv(os.path.join(self.fake_root_path, self.fake_data_path))
        
        # fake数据没有date列，创建虚拟的时间戳
        # 为每个样本创建索引作为时间戳
        fake_df['date'] = pd.date_range(start='2000-01-01', periods=len(fake_df), freq=self.freq)

        cols = list(fake_df.columns)
        cols.remove(self.target)
        cols.remove('date')
        fake_df = fake_df[['date'] + cols + [self.target]]
        
        if self.features == 'M' or self.features == 'MS':
            cols_data = fake_df.columns[1:]
            fake_data = fake_df[cols_data]
        else:  
            cols_data = [col for col in fake_df.columns if col != 'date']
            fake_data = fake_df[[self.target]]
        # 缩放处理
        if self.scale:
            fake_data = self.scaler.transform(fake_data.values)
        else:
            fake_data = fake_data.values
        
        df_stamp = fake_df[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            fake_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            fake_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            fake_stamp = fake_stamp.transpose(1, 0)
        
        return fake_data, fake_stamp

        
    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            # print(self.scaler.mean_)
            # exit()
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        
        if self.fake is not None and self.set_type == 0:
            # 只在训练集添加fake数据
            fake_data, fake_stamp = self.__read_fake_data__()
            
            # 确保fake数据长度足够
            fake_sample_len = self.seq_len + self.pred_len
            num_fake_samples = len(fake_data) // fake_sample_len
            
            if num_fake_samples > 0:
                
                # 提取fake样本
                fake_data_x_list = []
                fake_data_y_list = []
                fake_stamp_x_list = []
                fake_stamp_y_list = []
                
                for idx in range(num_fake_samples):
                    s_begin = idx * fake_sample_len
                    s_end = s_begin + self.seq_len
                    r_begin = s_end - self.label_len
                    r_end = r_begin + self.label_len + self.pred_len

                    fake_data_x_list.append(fake_data[s_begin:s_end])
                    fake_data_y_list.append(fake_data[r_begin:r_end])
                    fake_stamp_x_list.append(fake_stamp[s_begin:s_end])
                    fake_stamp_y_list.append(fake_stamp[r_begin:r_end])
                
                # 存储fake样本和权重
                self.fake_samples = {
                    'x': fake_data_x_list,
                    'y': fake_data_y_list,
                    'x_mark': fake_stamp_x_list,
                    'y_mark': fake_stamp_y_list
                }
                self.num_fake_samples = len(fake_data_x_list)
            else:
                self.num_fake_samples = 0
        else:
            self.num_fake_samples = 0
                    
    def __getitem__(self, index):
        if self.fake is not None and self.set_type == 0 and self.num_fake_samples > 0:
            total_original = len(self.data_x) - self.seq_len - self.pred_len + 1
            total_fake_expected = int(total_original * self.fake_weight)
            
            if index < total_fake_expected:
                fake_idx = index % self.num_fake_samples
                seq_x = self.fake_samples['x'][fake_idx]
                seq_y = self.fake_samples['y'][fake_idx]
                seq_x_mark = self.fake_samples['x_mark'][fake_idx]
                seq_y_mark = self.fake_samples['y_mark'][fake_idx]
                return seq_x, seq_y, seq_x_mark, seq_y_mark
            else:
                original_idx = index - total_fake_expected
                s_begin = original_idx
                s_end = s_begin + self.seq_len
                r_begin = s_end - self.label_len
                r_end = r_begin + self.label_len + self.pred_len
                
                seq_x = self.data_x[s_begin:s_end]
                seq_y = self.data_y[r_begin:r_end]
                seq_x_mark = self.data_stamp[s_begin:s_end]
                seq_y_mark = self.data_stamp[r_begin:r_end]
                return seq_x, seq_y, seq_x_mark, seq_y_mark
        
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
    
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
    
        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        original_len = len(self.data_x) - self.seq_len - self.pred_len + 1
        
        if self.fake is not None and self.set_type == 0 and self.num_fake_samples > 0:
            num_fake_to_add = int(original_len * self.fake_weight)
            return original_len + num_fake_to_add
        else:
            return original_len

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)