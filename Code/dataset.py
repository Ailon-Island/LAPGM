import numpy as np

from jittor.dataset import Dataset

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
        cls = data_list[0]['cls']

        return imgs, kpts, As, tgt, id_combination, cls







