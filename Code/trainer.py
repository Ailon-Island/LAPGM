import os
import time
from tqdm import tqdm, trange

import tensorboardX
import numpy as np

import jittor

from utils import plot_training_procedure, cal_metrics, scheduler_step


class Trainer:
    def __init__(self, opt, model, optimizer, lr_scheduler, test_loader, test_benchmark, loader=None, benchmark=None):
        self.opt = opt
        # self.total_iters = total_iters
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.total_epochs, self.epoch_iters, self.total_iters = 0, 0, 0
        self.training = False
        self.loader = loader
        self.benchmark = benchmark
        self.classes = getattr(benchmark, 'classes', [])
        self.test_loader = test_loader
        self.test_benchmark = test_benchmark
        self.test_classes = getattr(test_benchmark, 'classes', [])
        self.log_file = os.path.join(opt.checkpoint, opt.name, 'log.txt')
        self.writer = None

        if opt.resume:
            self.load(opt.resume_name, opt.resume_iter, opt.resume_model_only)

    def train_one_batch(self, batch_idx, img, kpt, A, tgt):
        # for fear of empty graph
        if kpt.shape[-1] == 0:
            self.total_iters += img.shape[0]
            metrics = cal_metrics(tgt, tgt)
            metrics = {name: 1. for name in metrics}
            return tgt.numpy(), tgt.numpy(), 0., metrics

        pred, pred_matching, loss = self.model(img, kpt, A, tgt, extractor_train=self.opt.extractor_train)
        self.optimizer.backward(loss)
        self.optimizer.step()
        self.optimizer.zero_grad()
        scheduler_step(self.lr_scheduler, img.shape[0])

        metrics = cal_metrics(pred_matching, tgt)

        self.total_iters += img.shape[0]

        return pred.numpy(), pred_matching.numpy(), loss.item(),  metrics

    def train_one_epoch(self, epoch_idx):
        self.model.train()

        num_iter = 0
        pbar_loader = tqdm(enumerate(self.loader, start=1), total=self.loader.__batch_len__())
        for batch_idx, (img, kpt, A, tgt, _, _) in pbar_loader:
            # skip batch/iterations if already trained
            if num_iter + img.shape[0] <= self.epoch_iters:
                continue
            elif num_iter < self.epoch_iters:
                trained_iters = self.epoch_iters - num_iter
                img, kpt, A, tgt = img[trained_iters:], kpt[trained_iters:], A[trained_iters:], tgt[trained_iters:]

            pred, pred_matching, loss, metrics = self.train_one_batch(batch_idx, img, kpt, A, tgt)

            # track training procedure
            num_iter += img.shape[0]
            if self.opt.use_tensorboard:
                self.writer.add_scalar('train/loss', loss, self.total_iters)
                for name, metric in metrics.items():
                    self.writer.add_scalar(f'train/{name}', metric, self.total_iters)

            # print training info
            pbar_loader.set_description(f'[TRAIN] [Epoch: {epoch_idx} Iteration: {self.total_iters} LR: {self.optimizer.lr:.6f}]')
            pbar_loader.set_postfix(
                loss=loss,
                **{name: f'{metric:.6f}' for name, metric in metrics.items()}
            )
            if num_iter % self.opt.train_log_iter == 0:
                self.log(
                    '[TRAIN] [Epoch: {} ({}/{} {:.0f}%) Iteration: {} LR: {:.6f}] Loss: {:.6f} '.format(
                        epoch_idx, num_iter, len(self.loader), 100. * num_iter / len(self.loader),
                        self.total_iters, self.optimizer.lr, loss,
                    ) +
                    ''.join(
                        [f'{name.capitalize()}: {metric:.6f} ' for name, metric in metrics.items()]
                    )
                )

        # finish training one epoch
        self.log(
            '[TRAIN] [Epoch: {} (finished) Iteration: {}] Loss: {:.6f} '.format(
                epoch_idx, self.total_iters, loss,
            ) +
            ''.join(
                [f'{name.capitalize()}: {metric:.6f} ' for name, metric in metrics.items()]
            )
        )

        # reset epoch_iters
        self.epoch_iters = 0

    def test_one_batch(self, batch_idx, img, kpt, A, tgt):
        with jittor.no_grad():
            pred, pred_matching, loss = self.model(img, kpt, A, tgt)

        metrics = cal_metrics(pred_matching, tgt)

        return pred.numpy(), pred_matching.numpy(), loss.item(), metrics

    def train(self, end_epoch=None):
        """
        Train the model till end_epoch or max_iters
        """
        training_saved = self.training
        self.training = True
        # check if training accessible
        if self.opt.train and (self.loader is None or self.benchmark is None):
            raise ValueError('Trainin requires training loader and benchmark.')
        # set up tensorboard
        if self.opt.use_tensorboard:
            self.writer = tensorboardX.SummaryWriter(log_dir=os.path.join(self.opt.checkpoint, self.opt.name, 'tensorboard'))

        # test now
        if self.opt.test_now:
            self.test()

        # start training
        self.log('[INFO] Start Training...')
        train_time = 0
        start_epoch = self.total_epochs + 1
        end_epoch = end_epoch or self.opt.epochs
        for epoch_idx in range(start_epoch, end_epoch + 1):
            # train one epoch
            self.log(f'[INFO] Training... Epoch: {epoch_idx} Iteration: {self.total_iters}')
            epoch_start_time = time.time()
            self.train_one_epoch(epoch_idx)
            epoch_time = time.time() - epoch_start_time
            train_time += epoch_time
            self.total_epochs += 1

            # save model
            if epoch_idx % self.opt.save_epoch == 0:
                self.save()

            # test training result
            if epoch_idx % self.opt.test_epoch == 0:
                self.test()

            # visualization
            self.log(f'[INFO] End of epoch. Epoch: {epoch_idx} Iteration: {self.total_iters} Time: {epoch_time:.2f}s')
            if self.total_iters >= self.opt.max_iters:
                self.log('Reach max iteration. Stop training.')
                break

        # test final result
        if self.total_epochs == 0 or self.total_epochs % self.opt.test_epoch != 0:
            self.test()

        # save final training procedure
        self.writer.close()

        self.log(f'[INFO] End of training. Total epochs: {self.total_epochs} Total iterations: {self.total_iters} Total time: {train_time:.2f}s')

        self.training = training_saved

    def test(self):
        """
        Test the model on test dataset.
        """
        training_saved = self.training
        self.training = False
        self.log(f'[INFO] Testing... Epoch: {self.total_epochs} Iteration: {self.total_iters}')
        test_start_time = time.time()
        self.model.eval()

        total_loss = 0.
        total_metrics = None
        pred_matching_list, ids_list, cls_list = [], [], []
        num_iter = 0
        pbar_loader = tqdm(enumerate(self.test_loader, start=1), total=self.test_loader.__batch_len__())
        for batch_idx, (img, kpt, A, tgt, ids, cls) in pbar_loader:
            # fix batched list of strings
            ids = np.array(ids).T

            # forward
            pred, pred_matching, loss, metrics = self.test_one_batch(batch_idx, img, kpt, A, tgt)

            # track testing procedure
            total_loss += loss * img.shape[0]
            if total_metrics is None:
                total_metrics = {name: metric * img.shape[0] for name, metric in metrics.items()}
            else:
                for name, metric in metrics.items():
                    total_metrics[name] += metric * img.shape[0]
            pred_matching_list += [pred_matching]
            ids_list += [ids]
            cls_list += cls
            num_iter += img.shape[0]

            # print testing info
            pbar_loader.set_description(f'[TEST] [Epoch: {self.total_epochs} Iteration: {self.total_iters}]')
            pbar_loader.set_postfix(
                loss=total_loss / num_iter,
                **{name: metric / num_iter for name, metric in total_metrics.items()}
            )
            if num_iter % self.opt.test_log_iter == 0:
                self.log(
                    '[TEST] [Epoch: {} Iteration: {}] [{}/{} {:.0f}%] Loss: {:.6f} '.format(
                        self.total_epochs, self.total_iters,
                        num_iter, len(self.test_loader), 100. * num_iter / len(self.test_loader),
                        total_loss / num_iter,
                    ) +
                    ''.join([f'{name.capitalize()}: {metric / num_iter:.6f} ' for name, metric in total_metrics.items()])
                )

        # calculate mean metrics
        total_loss /= num_iter
        total_metrics = {name: metric / num_iter for name, metric in total_metrics.items()}
        pred_matching = np.concatenate(pred_matching_list, axis=0)
        ids = np.concatenate(ids_list, axis=0)
        cls = cls_list
        metrics = self.eval(pred_matching, ids, cls)

        # track testing procedure
        if self.opt.use_tensorboard and self.writer is not None:
            self.writer.add_scalar('test/loss', total_loss, self.total_iters)
            for c, metric in metrics.items():
                for name, m in metric.items():
                    scalar_tag = f'test/{name}' if c == 'mean' else f'test/{c}/{name}'
                    self.writer.add_scalar(scalar_tag, m, self.total_iters)

        test_time = time.time() - test_start_time
        # print testing result
        self.log('[TEST] [Epoch: {} Iteration: {} Time: {:.2f}s] Loss: {:.6f}'.format(
                self.total_epochs, self.total_iters, test_time, total_loss
            )
        )
        for c, metric in metrics.items():
            if c != 'mean':
                self.log(
                    f'[TEST] [{c}]\t' +
                    ''.join([f'{name.capitalize()}: {m:.6f} ' for name, m in metric.items()])
                )
        self.log(
            '[TEST] [Mean]\t' +
            ''.join([f'{name.capitalize()}: {m:.6f} ' for name, m in metrics['mean'].items()])
        )

        self.training = training_saved

    def eval(self, pred_matching, ids, cls):
        """
        Evaluate the result of the model.
        """
        if self.training:
            raise ValueError('Cannot evaluate the model during training.')

        pred = [{'perm_mat': np.array(mat), 'ids': id, 'cls': c} for mat, id, c in zip(pred_matching, ids, cls)]

        # evaluate
        # pick classes in cls
        metrics = self.test_benchmark.eval(pred, set(cls))
        self.test_benchmark.rm_gt_cache()

        return metrics

    def save(self, model_only=False):
        """
        Save 'latest' model, and, optionally, a copy name with total_iters
        """
        if not self.opt.save_latest_only:
            self._save(as_latest=False, model_only=model_only)
        self._save(model_only=model_only)
        self.log(f'[INFO] Checkpoint saved for iter {self.total_iters}')

    def _save(self, as_latest=True, model_only=False):
        """
        Save current model, optimizer, scheduler, and training progress
        :param as_latest: whether to save as 'latest'
        :param model_only: only to save the model
        """
        iter_label = 'latest' if as_latest else f'iter_{self.total_iters}'

        save_dir = os.path.join(self.opt.checkpoint, self.opt.name, 'models')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        model_path = os.path.join(save_dir, f'{iter_label}.pkl')
        self.model.save(model_path)
        if self.opt.verbose:
            self.log(f'[INFO] Model saved to {model_path}')

        if not model_only:
            optimizer_path = os.path.join(save_dir, f'{iter_label}_optim.pkl')
            scheduler_path = os.path.join(save_dir, f'{iter_label}_sched.pkl')
            progress_path = os.path.join(save_dir, f'{iter_label}_progress.txt')
            jittor.save(self.optimizer.state_dict(), optimizer_path)
            if self.opt.verbose:
                self.log(f'[INFO] Optimizer saved to {optimizer_path}')
            jittor.save(self.lr_scheduler, scheduler_path)
            if self.opt.verbose:
                self.log(f'[INFO] Scheduler saved to {scheduler_path}')
            self.log(f'[INFO] Scheduler saved to {scheduler_path}')
            with open(progress_path, 'w') as f:
                f.write(f'{self.total_epochs} {self.epoch_iters} {self.total_iters}')
            if self.opt.verbose:
                self.log(f'[INFO] Progress saved to {progress_path}')

    def load(self, name=None, iter_label='latest', model_only=False, tolerance=0):
        """
        Load model, optimizer states, scheduler states, and training progress from checkpoint
        :param name: experiment name to load from
        :param iter_label: 'latest' or a number, the iteration to load
        :param model_only: load model only
        :param tolerance:
            0 for strict,
            1 to proceed when model loaded only,
            2 to proceed when model and progress loaded,
            3 to proceed when model not loaded
        """
        iter_label = f'iter_{iter_label}' if iter_label != 'latest' else iter_label
        name = self.opt.name if name is None else name

        load_dir = os.path.join(self.opt.checkpoint, name, 'models')
        if not os.path.exists(load_dir):
            if self.opt.pretrain:
                self.log(f'[INFO] No latest checkpoint found. Using the pretrained model directly.')
                return
            elif name == self.opt.name and iter_label == 'latest':
                self.log(f'[INFO] No latest checkpoint found. Start training from scratch.')
                return
            elif tolerance >= 3:
                self.log(f'[WARN] No checkpoint found at {load_dir}. Proceed anyway.')
                return
            else:
                raise FileNotFoundError(f'Failed to find checkpoint: no such directory {load_dir}')

        # load model and progress
        model_path = os.path.join(load_dir, f'{iter_label}.pkl')
        if not os.path.exists(model_path):
            if self.opt.pretrain:
                self.log(f'[INFO] No latest checkpoint found. Using the pretrained model directly.')
                return
            elif name == self.opt.name and iter_label == 'latest':
                self.log(f'[INFO] No latest model found. Start training from scratch.')
                return
            elif tolerance >= 3:
                self.log(f'[WARN] No model found at {model_path}. Proceed anyway.')
                return
            else:
                raise FileNotFoundError(f'Failed to load model: no such file {model_path}')
        self.model.load(model_path)
        if self.opt.verbose:
            self.log(f'[INFO] Model loaded from {model_path}')

        # if not model_only, load optimizer and scheduler states
        if not model_only:
            optimizer_path = os.path.join(load_dir, f'{iter_label}_optim.pkl')
            scheduler_path = os.path.join(load_dir, f'{iter_label}_sched.pkl')
            if not os.path.exists(optimizer_path):
                if tolerance >= 2:
                    self.log(f'[WARN] No optimizer found at {optimizer_path}. Proceed anyway.')
                else:
                    raise FileNotFoundError(f'Failed to load optimizer: no such file {optimizer_path}')
            else:
                self.optimizer.load_state_dict(jittor.load(optimizer_path))
                if self.opt.verbose:
                    self.log(f'[INFO] Optimizer loaded from {optimizer_path}')

            if not os.path.exists(scheduler_path):
                if tolerance >= 2:
                    self.log(f'[WARN] No scheduler found at {scheduler_path}. Proceed anyway.')
                else:
                    raise FileNotFoundError(f'Failed to load scheduler: no such file {scheduler_path}')
            else:
                self.lr_scheduler = jittor.load(scheduler_path)
                if self.opt.verbose:
                    self.log(f'[INFO] Scheduler loaded from {scheduler_path}')

        # load training progress
        progress_path = os.path.join(load_dir, f'{iter_label}_progress.txt')
        if not os.path.exists(progress_path):
            if tolerance >= 1:
                self.log(f'[WARN] No proceed found at {progress_path}. Proceed anyway.')
            else:
                raise FileNotFoundError(f'Failed to load proceed: no such file {progress_path}')
        else:
            with open(progress_path, 'r') as f:
                self.total_epochs, self.epoch_iters, self.total_iters = map(int, f.read().split())
            if self.opt.verbose:
                self.log(f'[INFO] Proceed loaded from {progress_path}')

        self.log(f'[INFO] Checkpoint loaded from {load_dir}')

    def log(self, msg):
        tqdm.write(msg)
        with open(self.log_file, 'a') as f:
            f.write(msg + '\n')
