import torch.nn as nn
import models.norms as norms
import torch
import torch.nn.functional as F

#--- Dual attention modules from DAGAN ---
#
#   input : feature
#   output : attention feature

# Position-wise
#
# Input : CLADE-Resblock last output feature
# Output : attention tensor
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        try:
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            scale = torch.cat([avg_out, max_out], dim=1)
            scale = self.conv(scale)
            out = x * self.sigmoid(scale)
        except Exception as e:
            print(e)
            out = x
        return out

# Scale-wise
#
# Input : CLADE-Resblock last and next to last output feature
# Output : attention tensor
class ChannelAttention(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ChannelAttention, self).__init__()
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.sigmod = nn.Sigmoid()

    def forward(self, x1, x2):
        # high, low
        x = torch.cat([x1,x2],dim=1)
        x = self.global_pooling(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmod(x)
        x2 = x * x2
        res = x2 + x1
        return res

#--- The CLADE resblock ---
#
# Hyperparameters
#   fin : input channel
#   fout : output channel
#   opt : option
#
# Input : tensor
# Output : tensor
class ResnetBlock_with_CLADE(nn.Module):
    def __init__(self, opt, fin, fout):
        super().__init__()
        self.opt = opt
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)
        sp_norm = norms.get_spectral_norm(opt)
        add_channels = 1 if not opt.no_instance else 0 
        self.conv_0 = sp_norm(nn.Conv2d(fin+add_channels, fmiddle, kernel_size=3, padding=1))
        self.conv_1 = sp_norm(nn.Conv2d(fmiddle+add_channels, fout, kernel_size=3, padding=1))
        if self.learned_shortcut:
            self.conv_s = sp_norm(nn.Conv2d(fin+add_channels, fout, kernel_size=1, bias=False))

        clade_conditional_input_dims = opt.semantic_nc

        self.norm_0 = norms.CLADE(opt, fin, clade_conditional_input_dims)
        self.norm_1 = norms.CLADE(opt, fmiddle, clade_conditional_input_dims)
        if self.learned_shortcut:
            self.norm_s = norms.CLADE(opt, fin, clade_conditional_input_dims)
        self.activ = nn.LeakyReLU(0.2, True)

    def forward(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        dx = self.conv_0(self.activ(self.norm_0(x, seg)))
        dx = self.conv_1(self.activ(self.norm_1(dx, seg)))
        out = x_s + dx
        return out

#--- generator ---
#
#   Input : semantic sementation & instance segmentation
#   Output: img tensor
class OASIS_Generator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        sp_norm = norms.get_spectral_norm(opt)
        ch = opt.channels_G

        self.channels = [16*ch, 16*ch, 16*ch, 8*ch, 4*ch, 2*ch, 1*ch]
        self.init_W, self.init_H = self.compute_latent_vector_size(opt)
        self.up = nn.Upsample(scale_factor=2)
        self.down = nn.AvgPool2d(2)

        self.body = nn.ModuleList([])
        for i in range(len(self.channels)-1):
            self.body.append(ResnetBlock_with_CLADE(opt, self.channels[i], self.channels[i+1]))

        self.fc = nn.Conv2d(self.opt.semantic_nc + self.opt.z_dim, 16 * ch, 3, padding=1)

        self.conv_img = nn.Conv2d(self.channels[-1], 3, 3, padding=1)

        self.conv_64 = nn.Conv2d(128, 64, 3, padding=1)
        self.cab = ChannelAttention(128,64)
        self.spatialAttn = SpatialAttention()

    def compute_latent_vector_size(self, opt):
        w = opt.crop_size // (2**(opt.num_res_blocks-1))
        h = round(w / opt.aspect_ratio)
        return h, w

    def forward(self, input, z=None): #z:3D_noise

        seg = input
        if self.opt.gpu_ids != "-1":
            seg.cuda()
        dev = seg.get_device() if self.opt.gpu_ids != "-1" else "cpu"
        z = torch.randn(seg.size(0), self.opt.z_dim, dtype=torch.float32, device=dev)
        z = z.view(z.size(0), self.opt.z_dim, 1, 1)
        z = z.expand(z.size(0), self.opt.z_dim, seg.size(2), seg.size(3))
        segmap = torch.cat((z, seg), dim = 1)
        
        x = F.interpolate(segmap, size=(self.init_W, self.init_H))
        x = self.fc(x)

        for i in range(self.opt.num_res_blocks):
            x = self.body[i](x, seg)
            if i < self.opt.num_res_blocks-1:
                x = self.up(x)
            if i == 4:
                x_tmp = self.conv_64(x)

        x_channel = self.cab(x_tmp,x)
        x_spatial = self.spatialAttn(x)
        x = x_channel + x_spatial

        x = torch.tanh(self.conv_img(F.leaky_relu(x, 2e-1)))

        return x
