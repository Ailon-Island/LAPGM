from collections.abc import Sequence, Mapping

import numpy as np

import jittor
from jittor.dataset import Dataset
from jittor.nn import pad

from utils import delaunay_triangulation_opencv, delaunay_triangulation


class GMDataset(Dataset):
    def __init__(self, opt, benchmark):
        super(GMDataset, self).__init__()
        self.opt = opt
        self.get_data = benchmark.get_data
        self.id_combination, self.total_len = benchmark.get_id_combination()
        self.num_classes = len(self.id_combination)
        self.class_len = [len(cmb) for cmb in self.id_combination]
        self.class_offset = [0] + list(np.cumsum(self.class_len))[:-1]
        self.train = benchmark.sets == 'train'
        self.shuffle = opt.data_shuffle and self.train
        self.batch_size = opt.batch_size

        # set_attrs must be called to set batch size total len and shuffle like __len__ function in pytorch
        self.set_attrs(batch_size=self.batch_size, total_len=self.total_len,
                       shuffle=self.shuffle)  # bs , total_len, shuffle

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        class_idx, cmb_idx = next(
            ((self.num_classes - index - 1, idx - offset) for index, offset in
             enumerate(reversed(self.class_offset)) if offset <= idx),
            None
        )
        id_combination = self.id_combination[class_idx][cmb_idx]
        kpt_shuffle = self.opt.kpt_shuffle if self.train else self.opt.kpt_shuffle_test
        data_list, perm_mat_dict, _ = self.get_data(list(id_combination), shuffle=kpt_shuffle)

        imgs, kpts = np.array([data['img'] for data in data_list], dtype=np.float32)/256, np.array([data['kpts'] for data in data_list])
        imgs = imgs.transpose(0, 3, 1, 2)
        kpts = np.array([
            [[kp['x'] for kp in kpt], [kp['y'] for kp in kpt]]
            for kpt in kpts
        ])
        As = np.array([delaunay_triangulation(kpt) for kpt in kpts])
        tgt = perm_mat_dict[(0, 1)].toarray()
        valid_mask = np.ones_like(tgt, dtype=bool)
        cls = data_list[0]['cls']

        item = {
            'imgs': imgs,  # (G, C, H, W)
            'kpts': kpts,  # (G, 2, N)
            'As': As,  # (G, N, N)
            'tgt': tgt,  # (N, N)
            'valid_mask': valid_mask,  # (N, N)
            'ids': id_combination,  # (G)
            'cls': cls,
        }

        return item

    @classmethod
    def collate_batch(cls, batch, key=None):
        r"""Puts each data field into a tensor with outer dimension batch size"""
        real_size = len(batch)
        elem = batch[0]
        elem_type = type(elem)
        if key == 'ids':
            # fix batched list of strings
            return np.array(batch).T
        if key in ['kpts', 'As', 'tgt', 'valid_mask']:
            elem_len = max([e.shape[-1] for e in batch])  # N
            if key == 'kpts':
                temp_data = np.zeros((real_size, 2, 2, elem_len), dtype=elem.dtype)
                for i, e in enumerate(batch):
                    temp_data[i, :, :, :e.shape[-1]] = e
            elif key == 'As':
                temp_data = np.zeros((real_size, 2, elem_len, elem_len), dtype=elem.dtype)
                for i, e in enumerate(batch):
                    temp_data[i, :, :e.shape[-1], :e.shape[-1]] = e
            elif key == 'tgt' or key == 'valid_mask':
                temp_data = np.zeros((real_size, elem_len, elem_len), dtype=elem.dtype)
                for i, e in enumerate(batch):
                    temp_data[i, :e.shape[-1], :e.shape[-1]] = e
            return temp_data

        if isinstance(elem, jittor.Var):
            temp_data = jittor.stack([data for data in batch], 0)
            return temp_data
        if elem_type is np.ndarray:
            temp_data = np.stack([data for data in batch], 0)
            return temp_data
        elif np.issubdtype(elem_type, np.integer):
            return np.int32(batch)
        elif isinstance(elem, int):
            return np.int32(batch)
        elif isinstance(elem, float):
            return np.float32(batch)
        elif isinstance(elem, str):
            return batch
        elif isinstance(elem, Mapping):
            return {key: cls.collate_batch([d[key] for d in batch], key) for key in elem}
        elif isinstance(elem, tuple):
            transposed = zip(*batch)
            return tuple(cls.collate_batch(samples) for samples in transposed)
        elif isinstance(elem, Sequence):
            transposed = zip(*batch)
            return [cls.collate_batch(samples) for samples in transposed]
        elif isinstance(elem, Image.Image):
            temp_data = np.stack([np.array(data) for data in batch], 0)
            return temp_data
        else:
            raise TypeError(f"Not support type <{elem_type.__name__}>")

    def to_jittor(self, batch):
        '''
        Change batch data to jittor array, such as np.ndarray, int, and float.
        '''
        if self.keep_numpy_array: return batch
        if isinstance(batch, jittor.Var): return batch
        to_jt = lambda x: jittor.array(x).stop_grad() \
            if self.stop_grad else jittor.array(x)
        if isinstance(batch, np.ndarray):
            # support for str numpy, just keep as it is
            if isinstance(batch.flatten()[0], str):
                return batch
            else:
                return to_jt(batch)
        if isinstance(batch, dict):
            new_batch = {}
            for k, v in batch.items():
                new_batch[k] = self.to_jittor(v)
            return new_batch
        if not isinstance(batch, (list, tuple)):
            return batch
        new_batch = []
        for a in batch:
            if isinstance(a, np.ndarray):
                new_batch.append(to_jt(a))
            else:
                new_batch.append(self.to_jittor(a))
        return new_batch




