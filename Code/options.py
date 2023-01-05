import os
import argparse


class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='NNGM Evaluation')
        self.initialized = False

    def initialize(self):
        # Basic
        self.parser.add_argument('--name', type=str, default='trial', help='name of experiment')
        self.parser.add_argument('--force_cpu', action='store_true', help='force to use cpu')

        # Data
        self.parser.add_argument('--dataset', type=str, default='WillowObject', help='dataset name')
        self.parser.add_argument('--obj_resize', type=tuple, default=(256, 256), help='resize object image')
        self.parser.add_argument('--data_shuffle', type=bool, default=True, help='shuffle data')
        self.parser.add_argument('--no_data_shuffle', action='store_false', dest='data_shuffle')
        self.parser.add_argument('--kpt_shuffle', type=bool, default=True, help='shuffle keypoints')
        self.parser.add_argument('--no_kpt_shuffle', action='store_false', dest='kpt_shuffle')
        self.parser.add_argument('--kpt_shuffle_test', type=bool, default=True, help='shuffle keypoints for test')
        self.parser.add_argument('--no_kpt_shuffle_test', action='store_false', dest='kpt_shuffle_test')

        # Model
        self.parser.add_argument('--model', type=str, default='pca-gm', help='matching method')
        self.parser.add_argument('--hungarian_attention', action='store_true', help='use hungarian attention')

        # Training
        self.parser.add_argument('--train', type=bool, default=True, help='train or test')
        self.parser.add_argument('--extractor_train', action='store_true', help='train extractor')
        self.parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
        self.parser.add_argument('--max_iters', type=int, default=9500, help='maximum number of iterations')
        self.parser.add_argument('--batch_size', type=int, default=32, help='batch size')
        self.parser.add_argument('--optimizer', type=str, default='adamw', help='optimizer, adam, adamw, or sgd')
        self.parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
        self.parser.add_argument('--weight_decay', type=float, default=0., help='weight decay')
        self.parser.add_argument('--lr_scheduler', type=str, default='step', help='learning rate scheduler, none, step, or cosine')
        self.parser.add_argument('--lr_schedule_iter', type=int, default=500, help='learning rate schedule iter')
        self.parser.add_argument('--lr_schedule_gamma', type=float, default=0.8, help='learning rate schedule gamma')

        # Testing
        self.parser.add_argument('--test_now', action='store_true', help='test at the beginning')
        self.parser.add_argument('--test_only', action='store_false', dest='train', help='test only')
        self.parser.add_argument('--test_epoch', type=int, default=1, help='test every n epochs')
        self.parser.add_argument('--test_final_only', action='store_true', help='test only final epoch')

        # Checkpoints
        self.parser.add_argument('--checkpoint', type=str, default='checkpoints', help='save directory')
        self.parser.add_argument('--save_epoch', type=int, default=1, help='save frequency')
        self.parser.add_argument('--save_latest_only', action='store_true', help='save latest only')
        self.parser.add_argument('--pretrain', type=str, default='', help='pretrained dataset name, will be loaded with pygmtools')
        self.parser.add_argument('--resume', type=bool, default=True, help='resume training')
        self.parser.add_argument('--no_resume', action='store_false', dest='resume')
        self.parser.add_argument('--resume_name', type=str, default=None, help='name of the experiment to resume from')
        self.parser.add_argument('--resume_iter', type=str, default='latest', help='resume iteration, \'\' for latest')
        self.parser.add_argument('--resume_model_only', action='store_true', help='resume model only')

        # Visualization
        self.parser.add_argument('--verbose', type=bool, default=False, help='verbose logging')
        self.parser.add_argument('--train_log_iter', type=int, default=100, help='train log frequency')
        self.parser.add_argument('--test_log_iter', type=int, default=1000, help='test log frequency')
        self.parser.add_argument('--use_tensorboard', type=bool, default=True, help='use tensorboard')
        self.parser.add_argument('--no_tesnorboard', action='store_false', dest='use_tensorboard')

        # Misc
        self.parser.add_argument('--seed', type=int, default=0, help='random seed')
        self.parser.add_argument('--debug', type=bool, default=False, help='debug mode')

        self.initialized = True

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        if self.opt.pretrain == '':
            self.opt.pretrain = False
        if self.opt.resume_iter == '':
            self.opt.resume_iter = 'latest'

        if self.opt.test_final_only:
            self.opt.test_epoch = self.opt.epochs

        if not self.opt.train:
            self.opt.resume_model_only = True

        if self.opt.debug:
            self.opt.name = 'DEBUG_' + self.opt.name
            self.opt.epochs = 1
            self.opt.batch_size = 1
            self.opt.save_epoch = 1
            self.opt.verbose = True

        args = vars(self.opt)
        print('------------ self.options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoint, self.opt.name)
        if not os.path.exists(expr_dir):
            os.makedirs(expr_dir)
        if self.opt.train and save:
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as self.opt_file:
                self.opt_file.write('------------ self.options -------------\n')
                for k, v in sorted(args.items()):
                    self.opt_file.write('%s: %s\n' % (str(k), str(v)))
                self.opt_file.write('-------------- End ----------------\n')

        return self.opt




