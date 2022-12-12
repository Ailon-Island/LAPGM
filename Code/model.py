import os

import numpy as np

import jittor
from jittor.nn import Module, interpolate
from jittor.models import vgg16_bn
import pygmtools as pygm

from utils import l2norm


class Extractor(Module):
    def __init__(self, vgg=None):
        super(Extractor, self).__init__()
        if vgg is None:
            vgg = vgg16_bn(pretrained=True)
        self.node_layers = jittor.nn.Sequential(*[_ for _ in list(vgg.features)[:31]])
        self.edge_layers = jittor.nn.Sequential(*[_ for _ in list(vgg.features)[31:38]])

        path = pygm.utils.download('vgg16_pca_voc_jittor.pt',
                                   'https://drive.google.com/u/0/uc?export=download&confirm=Z-AR&id=1qLxjcVq7X3brylxRJvELCbtCzfuXQ24J')
        self.load_state_dict(jittor.load(path))

    def execute(self, img):
        B, G = img.shape[:2]
        img_size = img.shape[-2:]
        img = img.reshape(-1, *img.shape[2:])
        feat_local = self.node_layers(img)
        feat_global = self.edge_layers(feat_local)
        feat_local = l2norm(feat_local)
        feat_global = l2norm(feat_global)

        feat_local_up = interpolate(feat_local, size=img_size, mode='bilinear')
        feat_global_up = interpolate(feat_global, size=img_size, mode='bilinear')

        feat = jittor.concat((feat_local_up, feat_global_up), dim=1)

        feat = feat.reshape(B, G, *feat.shape[1:])

        return feat


class LAPGM(Module):
    def __init__(self, opt, vgg=None):
        super(LAPGM, self).__init__()
        self.opt = opt
        self.extractor = Extractor(vgg)
        self.extractor.eval()
        # dummy data to initialize the model
        img = jittor.zeros([1, 1, 3, *opt.obj_resize])
        with jittor.no_grad():
            feat = self.extractor(img)
        F = feat.shape[-3]
        node = jittor.zeros([1, 1, F])
        A = jittor.zeros([1, 1, 1])
        if opt.model == 'pca-gm':
            self.gm_fn = pygm.pca_gm
        elif opt.model == 'ipca-gm':
            self.gm_fn = pygm.ipca_gm
        else:
            raise NotImplementedError(f'Model {opt.model} is not implemented')
        _, self.gm_model = pygm.pca_gm(node, node, A, A, return_network=True, pretrain=False)

    def execute(self, img, kpt, A, tgt=None, extractor_train=False):
        """
        Forward pass
        :param img: images, shape (B, G, C, H, W)
        :param kpt: keypoints, shape (B, G, 2, N)
        :param A: adjacency matrices, shape (B, N, N)
        :param tgt: target matching, shape (B, N, N)
        :param extractor_train: whether to train the extractor
        :return:
            pred: predicted soft matching, shape (B, N, N)
            pred_matching: predicted matching, shape (B, N, N)
            loss: loss
        """
        # get shape
        B, G = img.shape[0], img.shape[1]

        # extract features
        if self.train() and extractor_train:
            self.extractor.train()
            scope = jittor.enable_grad
        else:
            self.extractor.eval()
            scope = jittor.no_grad
        with scope():
            feat = self.extractor(img)  # B, G, F, H, W

        # get node features
        feat = feat.permute(0, 1, 3, 4, 2)  # B, G, H, W, F
        feat = feat.reshape(-1, *feat.shape[2:])  # BG, H, W, F
        rounded_kpt = jittor.round(kpt).long()  # B, G, 2, N
        rounded_kpt = rounded_kpt.permute(0, 1, 3, 2)  # B, G, N, 2
        rounded_kpt = rounded_kpt.reshape(-1, *rounded_kpt.shape[2:])  # BG, N, 2
        node = feat.gather(1, rounded_kpt[:, :, 0])  # BG, N, H, F
        node = node.gather(2, rounded_kpt[:, :, 1, None])  # BG, N, 1, F
        node = node.reshape(B, G, -1, node.shape[-1])  # B, G, N, F

        # match graphs
        pred, pred_matching = self.gm(node, A)  # B, N, N

        # compute loss
        if tgt is not None:
            loss = pygm.utils.permutation_loss(pred_matching, tgt)
            return pred, pred_matching, loss

        return pred, pred_matching

    def gm(self, node, A):
        """
        Match two graphs
        :param node: node features of the two graphs, shape (B, G, N, F)
        :param A: adjacency matrices of the two graphs, shape (B, G, N, N)
        :return:
            pred: predicted soft matching, shape (B, N, N)
            pred_matching: predicted matching, shape (B, N, N)
        """
        pred = self.gm_fn(node[:, 0], node[:, 1], A[:, 0], A[:, 1], network=self.gm_model)
        with jittor.no_grad():
            pred_matching = pygm.hungarian(pred)

        return pred, pred_matching


