from models.sync_batchnorm import DataParallelWithCallback
import models.generator as generators
import models.discriminator as discriminators
from models.diffaugment import DiffAugment
import os
import torch
import torch.nn as nn
from torch.nn import init
import models.losses as losses
from torchsummary import summary
import numpy.random as random
from ptflops import get_model_complexity_info

# Our GAN model that combine generator and discriminator, also compute loss
class OASIS_model(nn.Module):
    def __init__(self, opt):
        super(OASIS_model, self).__init__()
        self.opt = opt
        self.netG = generators.OASIS_Generator(opt)
        if opt.phase == "train":
            self.netD = discriminators.OASIS_Discriminator(opt)
        self.print_parameter_count()
        self.init_networks()
        self.load_checkpoints()

    def forward(self, image, input, mode, losses_computer, DA_p):
        # Branching is applied to be compatible with DataParallel

        # update generator
        if mode == "losses_G":

            fake = self.netG(input)
            output_D= self.netD(DiffAugment(fake, use_DA=random.choice([False, True],p=DA_p)))
            loss_G_adv = losses_computer.loss(output_D, input[:,:-1,:,:], for_real=True)
            loss_G = loss_G_adv

            return loss_G, [loss_G] 

        # update discriminator
        if mode == "losses_D":

            with torch.no_grad():
                fake = self.netG(input)

            loss_D = 0
            
            # cross entropy loss
            output_D_fake = self.netD(DiffAugment(fake, use_DA=random.choice([False, True],p=DA_p)))
            loss_D_fake = losses_computer.loss(output_D_fake, input[:,:-1,:,:], for_real=False)
            loss_D += loss_D_fake

            output_D_real = self.netD(DiffAugment(image, use_DA=random.choice([False, True],p=DA_p)))
            loss_D_real = losses_computer.loss(output_D_real, input[:,:-1,:,:], for_real=True)
            loss_D += loss_D_real

            # labelmix loss
            if not self.opt.no_labelmix:
                mixed_inp, mask = generate_labelmix(input[:,:-1,:,:], fake, image)
                output_D_mixed = self.netD(mixed_inp)
                loss_D_lm = self.opt.lambda_labelmix * losses_computer.loss_labelmix(mask, output_D_mixed, output_D_fake, output_D_real)
                loss_D += loss_D_lm

            return loss_D, [loss_D_fake, loss_D_real, loss_D_lm]

        #generate image from generator
        if mode == "generate":
            with torch.no_grad():
                fake = self.netG(input)
            return fake

    # load checkpoints
    def load_checkpoints(self):

        def load_part_weights(model, path):
            pretrained_dict = torch.load(path)
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

        if self.opt.phase == "test":
            which_iter = self.opt.ckpt_iter
            path = os.path.join(self.opt.checkpoints_dir, self.opt.name, "models", str(which_iter) + "_")
            load_part_weights(self.netG, path + "G.pth")
            load_part_weights(self.netG, path + "G.pth")

        elif self.opt.continue_train:
            which_iter = self.opt.which_iter
            path = os.path.join(self.opt.checkpoints_dir, self.opt.name, "models", str(which_iter) + "_")
            load_part_weights(self.netG, path + "G.pth")
            load_part_weights(self.netD, path + "D.pth")

    # print parameters and computation
    def print_parameter_count(self):
        if self.opt.phase == "train":
            summary(self.netG.cuda(),(self.opt.semantic_nc , int(self.opt.crop_size/2), self.opt.crop_size))
            summary(self.netD.cuda(),(3, int(self.opt.crop_size/2), self.opt.crop_size))
            macs, params = get_model_complexity_info(self.netG.cuda(), (self.opt.semantic_nc, int(self.opt.crop_size/2), self.opt.crop_size),
                as_strings=True,print_per_layer_stat=True, verbose=True)
            print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        else:
            summary(self.netG.cuda(),(self.opt.semantic_nc, int(self.opt.crop_size/2), self.opt.crop_size))
            macs, params = get_model_complexity_info(self.netG.cuda(), (self.opt.semantic_nc, int(self.opt.crop_size/2), self.opt.crop_size),
                as_strings=True,print_per_layer_stat=True, verbose=True)
            print('{:<30}  {:<8}'.format('Computational complexity: ', macs))

    # inint model weights
    def init_networks(self):
        def init_weights(m, gain=0.02):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:

                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):

                init.xavier_normal_(m.weight.data, gain=gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        if self.opt.phase == "train":
            networks = [self.netG, self.netD]
        else:
            networks = [self.netG]
        for net in networks:
            net.apply(init_weights)

def put_on_multi_gpus(model, opt):
    if opt.gpu_ids != "-1":
        gpus = list(map(int, opt.gpu_ids.split(",")))
        model = DataParallelWithCallback(model, device_ids=gpus).cuda()
    else:
        model.module = model
    assert len(opt.gpu_ids.split(",")) == 0 or opt.batch_size % len(opt.gpu_ids.split(",")) == 0
    return model

# prcoess image tensor to gpu device
def preprocess_input(opt, data):
    data['label'] = data['label'].long()
    if opt.gpu_ids != "-1":
        data['label'] = data['label'].cuda()
        data['instance'] = data['instance'].cuda()
        data['image'] = data['image'].cuda()

    label_map = data['label']
    bs, _, h, w = label_map.size()
    nc = opt.label_nc
    if opt.gpu_ids != "-1":
        input_label = torch.cuda.FloatTensor(bs, nc, h, w).zero_()
    else:
        input_label = torch.FloatTensor(bs, nc, h, w).zero_()
    input_semantics = input_label.scatter_(1, label_map, 1.0)

    inst_map = data['instance']
    instance_edge_map = get_edges(inst_map)
    input_semantics = torch.cat((input_semantics, instance_edge_map), dim=1)
    
    return data['image'], input_semantics

# combine fake foreground object and real backgorund object
def generate_labelmix(label, fake_image, real_image):
    target_map = torch.argmax(label, dim = 1, keepdim = True)
    all_classes = torch.unique(target_map)
    for c in all_classes:
        target_map[target_map == c] = torch.randint(0,2,(1,), device="cuda")
    target_map = target_map.float()
    mixed_image = target_map*real_image+(1-target_map)*fake_image
    return mixed_image, target_map

# get edge from instacne
def get_edges(t):
    ByteTensor = torch.cuda.ByteTensor
    edge = ByteTensor(t.size()).zero_()
    edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    return edge.float()