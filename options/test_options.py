'''
Author: Naiyuan liu
Github: https://github.com/NNNNAI
Date: 2021-11-23 17:03:58
LastEditors: Naiyuan liu
LastEditTime: 2021-11-23 17:08:08
Description: 
'''
from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        # Set attributes directly without using argparse
        self.ntest = float("inf")
        self.results_dir = './results/'
        self.aspect_ratio = 1.0
        self.phase = 'test'
        self.which_epoch = 'latest'
        self.how_many = 50
        self.cluster_path = 'features_clustered_010.npy'
        self.use_encoded_image = False
        self.export_onnx = None
        self.engine = None
        self.onnx = None
        self.Arc_path = 'arcface_model/arcface_checkpoint.tar'
        self.pic_a_path = 'D:/swap_data/ID/elon-musk-hero-image.jpeg'
        self.pic_b_path = './demo_file/multi_people.jpg'
        self.pic_specific_path = './crop_224/zrf.jpg'
        self.multispecific_dir = './demo_file/multispecific'
        self.video_path = 'G:/swap_data/video/HSB_Demo_Trim.mp4'
        self.temp_path = './temp_results'
        self.output_path = './output/'
        self.id_thres = 0.03
        self.no_simswaplogo = False
        self.use_mask = False
        self.crop_size = 512

        self.isTrain = False
        
        self.isTrain = False