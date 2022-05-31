import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from utils import check_exists, makedir_exist_ok, save, load
from .utils import make_classes_counts


class KDDCUP99(Dataset):
    data_name = 'KDDCUP99'

    def __init__(self, root, split):
        super().__init__()
        self.root = os.path.expanduser(root)
        self.split = split
        if not check_exists(os.path.join(self.processed_folder)):
            self.process()
        self.id, self.data, self.target = load(os.path.join(self.processed_folder, '{}.pt'.format(self.split)),
                                               mode='pickle')
        self.classes_counts = make_classes_counts(self.target)
        self.classes_to_labels, self.target_size = load(os.path.join(self.processed_folder, 'meta.pt'), mode='pickle')

    def __getitem__(self, index):
        id, data, target = torch.tensor(self.id[index]), torch.tensor(self.data[index]), torch.tensor(
            self.target[index])
        input = {'id': id, 'data': data, 'target': target}
        return input

    def __len__(self):
        return len(self.data)

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    def process(self):
        if not check_exists(self.raw_folder):
            self.download()
        train_set, test_set, meta = self.make_data()
        save(train_set, os.path.join(self.processed_folder, 'train.pt'), mode='pickle')
        save(test_set, os.path.join(self.processed_folder, 'test.pt'), mode='pickle')
        save(meta, os.path.join(self.processed_folder, 'meta.pt'), mode='pickle')
        return

    def download(self):
        makedir_exist_ok(self.raw_folder)
        return

    def __repr__(self):
        fmt_str = 'Dataset {}\nSize: {}\nRoot: {}\nnSplit: {}'.format(self.data_name, self.__len__(), self.root,
                                                                      self.split)
        return fmt_str

    def make_data(self):
        # https://github.com/timeamagyar/kdd-cup-99-python/blob/master/kdd%20preprocessing.ipynb
        from sklearn.preprocessing import LabelEncoder, Normalizer
        from sklearn.datasets import fetch_kddcup99
        dataset = fetch_kddcup99(as_frame=True)
        data, target = dataset['data'], dataset['target']
        data['label'] = target
        data = data.drop('num_outbound_cmds', axis=1)
        data = data.drop('is_host_login', axis=1)
        data['protocol_type'] = data['protocol_type'].astype('category')
        data['service'] = data['service'].astype('category')
        data['flag'] = data['flag'].astype('category')
        cat_columns = data.select_dtypes(['category']).columns
        data[cat_columns] = data[cat_columns].apply(lambda x: x.cat.codes)
        data = data.drop_duplicates(subset=None, keep='first')

        value_counts = data['label'].value_counts()
        other_index = value_counts.index[value_counts.values < 100]
        for i in range(len(other_index)):
            data.loc[data['label'] == other_index[i], 'label'] = b'unknown'

        le = LabelEncoder()
        data['label'] = le.fit_transform(data['label'])
        classes = le.classes_
        null_label = classes.tolist().index(b'normal.')
        normal_df = data.loc[data['label'] == null_label, :]
        abnormal_df = data.loc[data['label'] != null_label, :]
        train_df = normal_df.iloc[:-10000]
        valid_normal_df = normal_df.iloc[-10000:]
        test_df = pd.concat([valid_normal_df, abnormal_df])
        train_data, train_target = train_df.values[:, :39].astype(np.float32), train_df.values[:, 39].astype(np.int64)
        test_data, test_target = test_df.values[:, :39].astype(np.float32), test_df.values[:, 39].astype(np.int64)
        norm = Normalizer()
        train_data = norm.fit_transform(train_data)
        test_data = norm.transform(test_data)
        train_id, test_id = np.arange(len(train_data)).astype(np.int64), np.arange(len(test_data)).astype(np.int64)
        classes_to_labels = {classes[i]: i for i in range(len(classes))}
        target_size = len(classes_to_labels)
        return (train_id, train_data, train_target), (test_id, test_data, test_target), (
            classes_to_labels, target_size)
