import argparse
import os
from util import util
import torch

class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self):
        # experiment specifics
        self.name = 'people'
        self.gpu_ids = '0'
        self.checkpoints_dir = './checkpoints'
        self.norm = 'batch'
        self.use_dropout = False
        self.data_type = 32
        self.verbose = False
        self.fp16 = False
        self.local_rank = 0
        self.isTrain = True  # You mentioned self.isTrain = False, but in your original code, it's set to True

        # input/output sizes
        self.batchSize = 8
        self.loadSize = 1024
        self.fineSize = 512
        self.label_nc = 0
        self.input_nc = 3
        self.output_nc = 3

        # for setting inputs
        self.dataroot = './datasets/cityscapes/'
        self.resize_or_crop = 'scale_width'
        self.serial_batches = False
        self.no_flip = False
        self.nThreads = 2
        self.max_dataset_size = float("inf")

        # for displays
        self.display_winsize = 512
        self.tf_log = False

        # for generator
        self.netG = 'global'
        self.latent_size = 512
        self.ngf = 64
        self.n_downsample_global = 3
        self.n_blocks_global = 6
        self.n_blocks_local = 3
        self.n_local_enhancers = 1
        self.niter_fix_global = 0

        # for instance-wise features
        self.no_instance = False
        self.instance_feat = False
        self.label_feat = False
        self.feat_num = 3
        self.load_features = False
        self.n_downsample_E = 4
        self.nef = 16
        self.n_clusters = 10
        self.image_size = 224
        self.norm_G = 'spectralspadesyncbatch3x3'
        self.semantic_nc = 3
        self.initialized = True

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self
        self.opt.isTrain = self.isTrain   # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        
        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        if self.opt.isTrain:
            expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
            util.mkdirs(expr_dir)
            if save and not self.opt.continue_train:
                file_name = os.path.join(expr_dir, 'opt.txt')
                with open(file_name, 'wt') as opt_file:
                    opt_file.write('------------ Options -------------\n')
                    for k, v in sorted(args.items()):
                        opt_file.write('%s: %s\n' % (str(k), str(v)))
                    opt_file.write('-------------- End ----------------\n')
        return self.opt
