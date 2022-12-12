import os
import itertools
import matplotlib.pyplot as plt # for plotting
from matplotlib.patches import ConnectionPatch # for plotting matching result

import scipy.io as sio # for loading .mat file
import scipy.spatial as spa # for Delaunay triangulation
from sklearn.decomposition import PCA as PCAdimReduc

import numpy as np

import jittor
from jittor import Var, nn


# training
class DummyScheduler:
    def __init__(self):
        pass

    def step(self):
        pass


# calculating
def delaunay_triangulation(kpt):
    d = spa.Delaunay(kpt.transpose())
    A = np.zeros((len(kpt[0]), len(kpt[0])))
    for simplex in d.simplices:
        for pair in itertools.permutations(simplex, 2):
            A[pair] = 1
    return A


def local_response_norm(input: Var, size: int, alpha: float = 1e-4, beta: float = 0.75, k: float = 1.0) -> Var:
    """
    jittor implementation of local_response_norm
    """
    dim = input.ndim
    assert dim >= 3

    if input.numel() == 0:
        return input

    div = input.multiply(input).unsqueeze(1)
    if dim == 3:
        div = nn.pad(div, (0, 0, size // 2, (size - 1) // 2))
        div = nn.avg_pool2d(div, (size, 1), stride=1).squeeze(1)
    else:
        sizes = input.size()
        div = div.view(sizes[0], 1, sizes[1], sizes[2], -1)
        div = nn.pad(div, (0, 0, 0, 0, size // 2, (size - 1) // 2))
        div = nn.AvgPool3d((size, 1, 1), stride=1)(div).squeeze(1)
        div = div.view(sizes)
    div = div.multiply(alpha).add(k).pow(beta)
    return input / div


def l2norm(node_feat):
    return local_response_norm(
        node_feat, node_feat.shape[1] * 2, alpha=node_feat.shape[1] * 2, beta=0.5, k=0)


# plotting
def plot_image_with_graph(img, kpt, A=None, save_path=None):
    plt.imshow(img)
    plt.scatter(kpt[0], kpt[1], c='w', edgecolors='k')
    if A is not None:
        for idx in jittor.nonzero(A, as_tuple=False):
            plt.plot((kpt[0, idx[0]], kpt[0, idx[1]]), (kpt[1, idx[0]], kpt[1, idx[1]]), 'k-')

    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def cal_metrics(pred, gt):
    with jittor.no_grad():
        p = (pred * gt).sum(dims=[1, 2]) / pred.sum(dims=[1, 2])
        r = (pred * gt).sum(dims=[1, 2]) / gt.sum(dims=[1, 2])
        f1 = 2 * p * r / (p + r + 1e-8)

    metrics = {'precision': p.mean().item(), 'recall': r.mean().item(), 'f1': f1.mean().item()}
    return metrics


def plot_training_procedure(train_losses=None, train_accs=None, test_losses=None, test_accs=None, iters=None, epochs=None, save_dir=None):
        plt.figure(figsize=(10, 5))
        if train_losses is not None:
            plt.plot(iters, train_losses, label='Train loss')
        if test_losses is not None:
            plt.plot(epochs, test_losses, label='Test loss')
        plt.ylim(0, 2)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, 'loss.png'))
        plt.show()

        plt.figure(figsize=(10, 5))
        if train_accs is not None:
            plt.plot(iters, train_accs, label='Train accuracy')
        if test_accs is not None:
            plt.plot(epochs, test_accs, label='Test accuracy')
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.legend()
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, 'accuracy.png'))
        plt.show()

