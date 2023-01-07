import os

import numpy as np

import jittor
from jittor.nn import Module, interpolate
from jittor.models import vgg16_bn
import pygmtools as pygm

from utils import l2norm


def get_gm_model(model, pretrain=False):
    edge_feature = False
    if model == 'pca-gm':
        gm_fn = pygm.pca_gm
    elif model == 'ipca-gm':
        gm_fn = pygm.ipca_gm
    elif model == 'cie':
        gm_fn = pygm.cie
        edge_feature = True
    else:
        raise NotImplementedError(f'Model {model} is not implemented')
    gm_model = pygm.utils.get_network(gm_fn, pretrain=pretrain)

    return gm_fn, gm_model, edge_feature


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
        self.gm_fn, self.gm_model, self.edge_feature = get_gm_model(opt.model, pretrain=self.opt.pretrain)
        self.lambda_hungarian = 0. if not opt.hungarian_attention else opt.lambda_hungarian

    def execute(self, inputs, extractor_train=False):
        """
        Forward pass
        :param inputs: dict of inputs
            'imgs': images, shape (B, G, C, H, W)
            'kpts': keypoints, shape (B, G, 2, N)
            'As': adjacency matrices, shape (B, G, N, N)
            'tgt': target matching, shape (B, N, N) (optional)
            'valid_mask': valid mask for padding, shape (B, N, N) (optional)
        :param extractor_train: whether to train the extractor
        :return:
            pred: predicted soft matching, shape (B, N, N)
            pred_matching: predicted matching, shape (B, N, N)
            loss: loss
        """
        img, kpt, A, tgt, valid_mask = inputs['imgs'], inputs['kpts'], inputs['As'], inputs.get('tgt', None), inputs.get('valid_mask', None)

        # select scope up to extractor training mode
        if self.is_training() and extractor_train:
            self.extractor.train()
            scope = jittor.enable_grad
        else:
            self.extractor.eval()
            scope = jittor.no_grad

        with scope():
            # extract image features
            feat = self.extractor(img)  # B, G, F, H, W

            # get node features
            feat = feat.reshape(-1, *feat.shape[2:]).permute(0, 2, 3, 1)  # BG, H, W, F
            rounded_kpt = jittor.round(kpt).long()  # B, G, 2, N
            rounded_kpt = rounded_kpt.permute(0, 1, 3, 2)  # B, G, N, 2
            rounded_kpt = rounded_kpt.reshape(-1, *rounded_kpt.shape[2:])  # BG, N, 2
            node = feat.gather(1, rounded_kpt[:, :, 1])  # BG, N, H, F
            node = node.gather(2, rounded_kpt[:, :, 0, None])  # BG, N, 1, F
            node = node.reshape(*img.shape[:2], -1, node.shape[-1])  # B, G, N, F

            # get edge features
            Q = None
            if self.edge_feature:
                kpt_dis = kpt.unsqueeze(3) - kpt.unsqueeze(4)  # B, G, 2, N, N
                kpt_dis = jittor.norm(kpt_dis, p=2, dim=2)  # B, G, N, N
                Q = jittor.exp(-kpt_dis / imgs.shape[-1]).unsqueeze(-1).float32()  # B, G, N, N, 1

        # match graphs
        pred, pred_matching = self.gm(node, A, Q, valid_mask)  # B, N, N

        # compute loss
        if tgt is not None:
            if self.lambda_hungarian > 0:
                Z = jittor.bitwise_or(pred_matching, tgt).float32()
                Z = jittor.clamp(Z + self.lambda_hungarian, 0., 1.)
                pred.register_hook(lambda grad: grad * Z)
            loss = pygm.utils.permutation_loss(pred, tgt)
            return pred, pred_matching, loss

        return pred, pred_matching

    def gm(self, node, A, Q, valid_mask):
        """
        Match two graphs
        :param node: node features of the two graphs, shape (B, G, N, F)
        :param A: adjacency matrices of the two graphs, shape (B, G, N, N)
        :param Q: edge features of the two graphs, shape (B, G, N, N, 1)
        :param valid_mask: valid mask for padding, shape (B, N, N)
        :return:
            pred: predicted soft matching, shape (B, N, N)
            pred_matching: predicted matching, shape (B, N, N)
        """
        if self.edge_feature:
            pred = self.gm_fn(node[:, 0], node[:, 1], A[:, 0], A[:, 1], Q[:, 0], Q[:, 1], network=self.gm_model)
        else:
            pred = self.gm_fn(node[:, 0], node[:, 1], A[:, 0], A[:, 1], network=self.gm_model)

        if valid_mask is not None:
            pred[valid_mask == False] = 0.

        with jittor.no_grad():
            pred_matching = pygm.hungarian(pred)
            if valid_mask is not None:
                pred_matching[valid_mask == False] = 0.

        return pred, pred_matching






