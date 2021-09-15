import torch.nn.utils.spectral_norm as spectral_norm
from models.sync_batchnorm import SynchronizedBatchNorm2d
import torch.nn as nn
import torch.nn.functional as F
import torch

#--- Class Affine ---
#
#   From CLADE
#
#   Input : tensor
#   Output : class affine tensor
#
#   Each class will process with embedding layer

class ClassAffine(nn.Module):
    def __init__(self, label_nc, affine_nc):
        super(ClassAffine, self).__init__()
        self.affine_nc = affine_nc
        self.label_nc = label_nc
        self.weight = nn.Parameter(torch.Tensor(self.label_nc, self.affine_nc))
        self.bias = nn.Parameter(torch.Tensor(self.label_nc, self.affine_nc))
        nn.init.uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def affine_embed(self, mask):
        arg_mask = torch.argmax(mask, 1).long() # [n, h, w]
        class_weight = F.embedding(arg_mask, self.weight).permute(0, 3, 1, 2) # [n, c, h, w]
        class_bias = F.embedding(arg_mask, self.bias).permute(0, 3, 1, 2) # [n, c, h, w]
        return class_weight, class_bias

    def forward(self, input, mask):
        class_weight, class_bias = self.affine_embed(mask)
        x = input * class_weight + class_bias
        return x

#--- CLADE Normalization ---
#
#   From CLADE
#
#   Input : feature tensor
#   Output : Normalizaed tensor
#
#   method:
#       1.batch norm
#       2.class affine
#       3.cat instance information
#
#   Parameters:
#       norm_nc : input tensor channel
#       label_nc : input seg map channel

class CLADE(nn.Module):
    def __init__(self, opt, norm_nc, label_nc, no_instance=False):
        super().__init__()
        self.no_instance = no_instance
        self.param_free_norm = get_norm_layer(opt, norm_nc)
        self.class_specified_affine = ClassAffine(label_nc, norm_nc)
        if not no_instance:
            self.inst_conv = nn.Conv2d(1, 1, kernel_size=1, padding=0)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. scale the segmentation mask
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')

        if not self.no_instance:
            inst_map = torch.unsqueeze(segmap[:,-1,:,:],1)
            segmap = segmap[:,:-1,:,:]

        # Part 3. class affine
        out = self.class_specified_affine(normalized, segmap)

        if not self.no_instance:
            inst_feat = self.inst_conv(inst_map)
            out = torch.cat((out, inst_feat), dim=1)

        return out

def get_spectral_norm(opt):
    if opt.no_spectral_norm:
        return torch.nn.Identity()
    else:
        return spectral_norm

def get_norm_layer(opt, norm_nc):
    if opt.param_free_norm == 'instance':
        return nn.InstanceNorm2d(norm_nc, affine=False)
    if opt.param_free_norm == 'syncbatch':
        return SynchronizedBatchNorm2d(norm_nc, affine=False)
    if opt.param_free_norm == 'batch':
        return nn.BatchNorm2d(norm_nc, affine=False)
    if opt.param_free_norm == 'sandwich':
        return SynchronizedBatchNorm2d(norm_nc, affine=True)
    else:
        raise ValueError('%s is not a recognized param-free norm type in SPADE'
                         % opt.param_free_norm)