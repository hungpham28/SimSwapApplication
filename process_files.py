import cv2
import torch
import fractions
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from models.models import create_model
from options.test_options import TestOptions
from insightface_func.face_detect_crop_multi import Face_detect_crop
from util.videoswap import video_swap
from insightface_func.face_detect_crop_single import Face_detect_crop
from util.reverse2original import reverse2wholeimage
import os
from util.add_watermark import watermark_image
from util.norm import SpecificNorm
from parsing_model.model import BiSeNet
import uuid
import numpy as np
def lcm(a, b): return abs(a * b) / fractions.gcd(a, b) if a and b else 0


transformer_Arcface = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
transformer = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def swap_video(video_path, pic_a):
    opt = TestOptions().parse()

    start_epoch, epoch_iter = 1, 0
    crop_size = 224

    torch.nn.Module.dump_patches = True
    # if crop_size == 512:
    #     opt.which_epoch = 550000
    #     opt.name = '512'
    #     mode = 'ffhq'
    # else:
    #     mode = 'None'
    model = create_model(opt)
    model.eval()


    app = Face_detect_crop(name='antelope', root='./insightface_func/models')
    app.prepare(ctx_id= 0, det_thresh=0.6, det_size=(640,640),mode=None)
    with torch.no_grad():
        # pic_a = opt.pic_a_path
        # img_a = Image.open(pic_a).convert('RGB')
        img_a_whole = cv2.imread(pic_a)
        img_a_align_crop, _ = app.get(img_a_whole,crop_size)
        img_a_align_crop_pil = Image.fromarray(cv2.cvtColor(img_a_align_crop[0],cv2.COLOR_BGR2RGB))
        img_a = transformer_Arcface(img_a_align_crop_pil)
        img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])

        # convert numpy to tensor
        img_id = img_id.cuda()
        # img_att = img_att.cuda()

        #create latent id
        img_id_downsample = F.interpolate(img_id, size=(112,112))
        latend_id = model.netArc(img_id_downsample)
        latend_id = F.normalize(latend_id, p=2, dim=1)
        random_filename = str(uuid.uuid4())+".mp4"
        video_swap(video_path, latend_id, model, app, opt.output_path+random_filename,temp_results_dir=opt.temp_path,
                   no_simswaplogo=opt.no_simswaplogo,use_mask=opt.use_mask,crop_size=crop_size)
        return os.path.join(opt.output_path, random_filename)
def _totensor(array):
    tensor = torch.from_numpy(array)
    img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
    return img.float().div(255)

def swap_image(pic_a,pic_b):
    opt = TestOptions()

    opt.initialize()
    opt = opt.parse()
    opt.pic_a_path = pic_a
    opt.pic_b_path = pic_b
    opt.crop_size = 224
    start_epoch, epoch_iter = 1, 0

    torch.nn.Module.dump_patches = True

    model = create_model(opt)
    model.eval()

    with torch.no_grad():

        pic_a = opt.pic_a_path
        # pic_a = 'D:/swap_data/ID/elon-musk-hero-image.jpeg'
        img_a = Image.open(pic_a)

        img_a = transformer_Arcface(img_a)
        img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])

        pic_b = opt.pic_b_path
        # pic_b = 'D:/swap_data/ID/elon-musk-hero-image.jpeg'

        img_b = Image.open(pic_b)
        img_b = transformer(img_b)
        img_att = img_b.view(-1, img_b.shape[0], img_b.shape[1], img_b.shape[2])

        # convert numpy to tensor
        img_id = img_id.cuda()
        img_att = img_att.cuda()

        #create latent id
        img_id_downsample = F.interpolate(img_id, size=(112,112))
        latend_id = model.netArc(img_id_downsample)
        latend_id = latend_id.detach().to('cpu')
        latend_id = latend_id/np.linalg.norm(latend_id,axis=1,keepdims=True)
        latend_id = latend_id.to('cuda')


        ############## Forward Pass ######################
        img_fake = model(img_id, img_att, latend_id, latend_id, True)


        for i in range(img_id.shape[0]):
            if i == 0:
                row1 = img_id[i]
                row2 = img_att[i]
                row3 = img_fake[i]
            else:
                row1 = torch.cat([row1, img_id[i]], dim=2)
                row2 = torch.cat([row2, img_att[i]], dim=2)
                row3 = torch.cat([row3, img_fake[i]], dim=2)

        #full = torch.cat([row1, row2, row3], dim=1).detach()
        full = row3.detach()
        full = full.permute(1, 2, 0)
        output = full.to('cpu')
        output = np.array(output)
        output = output[..., ::-1]

        output = output*255


        random_filename = str(uuid.uuid4()) + '.jpg'
        cv2.imwrite(opt.output_path + random_filename, output)
        return os.path.join(opt.output_path, random_filename)