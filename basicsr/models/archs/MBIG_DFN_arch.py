import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.models.archs.arch_util import *
from basicsr.models.archs.local_arch import Local_Base
from basicsr.models.archs.win_util import *
from torchvision.ops import DeformConv2d


class SimpleGate(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class WFSIB(nn.Module):
    def __init__(self, dim, window_size_fft=-1, DW_Expand=2, FFN_Expand=1, sin=True, bias=False):
        super().__init__()
        self.sin = sin
        self.window_size_fft = window_size_fft
        self.norm1 = LayerNorm2d(dim)
        dw_channel = dim * DW_Expand
        self.body = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                      bias=bias),
            nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                      groups=dw_channel,
                      bias=bias)
        )
        self.sg = SimpleGate()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, stride=1, groups=1,
                      bias=bias),
        )
        self.end = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, stride=1,
                             groups=1, bias=bias)

        self.fft_block1 = fft_bench_complex_mlp(dim, DW_Expand, window_size=window_size_fft, bias=bias,
                                                act_method=nn.GELU)  # , act_method=nn.GELU

        self.norm2 = LayerNorm2d(dim)
        ffn_channel = FFN_Expand * dim
        self.linear = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(
            LayerNorm2d(dim),
            nn.Conv2d(in_channels=dim, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=bias),
            nn.Conv2d(in_channels=ffn_channel, out_channels=ffn_channel, kernel_size=3, padding=1, stride=1,
                      groups=ffn_channel, bias=bias)
        )
        self.gelu = nn.GELU()
        self.over = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, stride=1,
                              groups=1,
                              bias=bias)

        # self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        # self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        x_ = self.norm1(x)
        x = self.body(x_)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.end(x)

        top = inp + x
        bot = self.fft_block1(x_)

        short_cut = top
        top = self.norm2(top)
        top = top.permute(0, 2, 3, 1).contiguous()
        top = self.linear(top)
        top = top.permute(0, 3, 1, 2).contiguous()

        bot = self.mlp(bot)
        bot = self.gelu(bot)
        out = top * bot
        out = self.over(out)
        return short_cut + out


class GSCAB(nn.Module):
    def __init__(self, dim, DW_Expand=2, FFN_Expand=1, bias=False):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        dw_channel = dim * DW_Expand
        self.body = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                      bias=bias),
            # nn.PReLU(),
            nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                      groups=dw_channel,
                      bias=bias)
        )
        self.sg = SimpleGate()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, stride=1, groups=1,
                      bias=bias),
        )
        self.end = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, stride=1,
                             groups=1, bias=bias)

        self.norm2 = LayerNorm2d(dim)
        ffn_channel = FFN_Expand * dim
        self.linear = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(
            LayerNorm2d(dim),
            nn.Conv2d(in_channels=dim, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=bias),
            nn.Conv2d(in_channels=ffn_channel, out_channels=ffn_channel, kernel_size=3, padding=1, stride=1,
                      groups=ffn_channel, bias=bias)
        )
        self.gelu = nn.GELU()
        self.over = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, stride=1,
                              groups=1,
                              bias=bias)
        # self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        # self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        x_ = self.norm1(x)
        x = self.body(x_)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.end(x)

        top = inp + x
        bot = top

        short_cut = top
        top = self.norm2(top)
        top = top.permute(0, 2, 3, 1).contiguous()
        top = self.linear(top)
        top = top.permute(0, 3, 1, 2).contiguous()

        bot = self.mlp(bot)
        bot = self.gelu(bot)
        out = top * bot
        out = self.over(out)
        return short_cut + out


class FFGM(nn.Module):
    def __init__(self, dim, bias=False):
        super(FFGM, self).__init__()
        self.patch_size = 8
        self.dim = dim
        self.project_in = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.fft = nn.Parameter(torch.ones((dim, 1, 1, self.patch_size, self.patch_size // 2 + 1)))
        self.project_out = nn.Conv2d(dim * 2, dim * 2, kernel_size=1, bias=bias)
        self.sg = SimpleGate()
        self.inverse_conv = DeformConv2d(in_channels=dim, out_channels=dim, kernel_size=7, padding=3, bias=bias)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, bias=bias)

    def forward(self, enc, dec, inverse_offset, inverse_weight):
        _, _, h, w = enc.shape
        reverse = self.inverse_conv(input=enc, offset=inverse_offset, mask=inverse_weight)
        reverse = self.conv1(reverse)

        enc = self.project_in(enc)
        enc = check_image_size(enc, self.patch_size)
        x_patch = rearrange(enc, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        x_patch_fft = torch.fft.rfft2(x_patch.float())
        x_patch_fft = x_patch_fft * self.fft
        x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))
        enc = rearrange(x_patch, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                        patch2=self.patch_size)
        enc = enc[:, :, :h, :w]

        enc = enc + reverse
        enc = self.conv2(enc)

        feature = torch.cat([enc, dec], dim=1)
        feature = self.project_out(feature)
        feature = self.sg(feature)
        return feature


class Downsample(nn.Module):
    def __init__(self, n_feat, bias=False):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(n_feat, n_feat * 2, 3, stride=1, padding=1, bias=bias))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat, bias=False):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                  nn.Conv2d(n_feat, n_feat // 2, 3, stride=1, padding=1, bias=bias))

    def forward(self, x):
        return self.body(x)


class MBIG_DFN(nn.Module):
    def __init__(self, img_channel=3, width=48, enc_blk_nums=[2, 4, 18],
                 dec_blk_nums=[18, 4, 2], window_size_e_fft=[64, 32, 16], n_heads=3, combinate_heads=True, bias=False):
        super().__init__()
        window_size_d_fft = window_size_e_fft[::-1]  # [16, 32, 64]
        self.patch_embed = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1,
                                      groups=1,
                                      bias=bias)
        self.output = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1,
                                groups=1,
                                bias=bias)
        chan = width

        # Encoder
        self.encoder_level1 = nn.Sequential(*[
            WFSIB(chan, window_size_e_fft[0], bias=bias) for i in range(enc_blk_nums[0])])
        self.encoder_level2 = nn.Sequential(*[
            GSCAB(chan * 2 ** 1, bias=bias) for i in range(enc_blk_nums[1])])
        self.encoder_level3 = nn.Sequential(*[
            GSCAB(chan * 2 ** 2, bias=bias) for i in range(enc_blk_nums[2])])

        # Decoder
        self.decoder_level3 = nn.Sequential(*[
            GSCAB(chan * 2 ** 2, bias=bias) for i in range(dec_blk_nums[0])])
        self.decoder_level2 = nn.Sequential(*[
            GSCAB(chan * 2 ** 1, bias=bias) for i in range(dec_blk_nums[1])])
        self.decoder_level1 = nn.Sequential(*[
            WFSIB(chan, window_size_d_fft[2], bias=bias) for i in range(dec_blk_nums[2])])

        # Middle
        self.offset_conv1 = nn.Conv2d(in_channels=chan * 4, out_channels=50, kernel_size=3, padding=1, bias=bias)
        self.offset_conv2 = nn.Conv2d(in_channels=50, out_channels=98, kernel_size=3, padding=1, bias=bias)
        self.offset_conv3 = nn.Conv2d(in_channels=98, out_channels=98, kernel_size=3, padding=1, bias=bias)

        self.weight_conv1 = nn.Conv2d(in_channels=chan * 4, out_channels=25, kernel_size=3, padding=1, bias=bias)
        self.weight_activation1 = nn.Softmax(dim=1)
        self.weight_conv2 = nn.Conv2d(in_channels=25, out_channels=49, kernel_size=3, padding=1, bias=bias)
        self.weight_conv3 = nn.Conv2d(in_channels=49, out_channels=49, kernel_size=3, padding=1, bias=bias)

        self.inverse_conv = DeformConv2d(in_channels=chan * 4, out_channels=chan * 4, kernel_size=7, padding=3,
                                         bias=bias)

        self.sta3down1_2 = Downsample(chan, bias=bias)
        self.sta3down2_3 = Downsample(chan * 2 ** 1, bias=bias)
        self.sta3up3_2 = Upsample(chan * 2 ** 2, bias=bias)
        self.sta3up2_1 = Upsample(chan * 2 ** 1, bias=bias)
        self.ConvOut3_a = nn.Sequential(nn.PixelShuffle(4),  # 输入b*(4width)*(h/4)*(w/4) 输出b*(width/4)*h*w
                                        nn.Conv2d(in_channels=width // 4, out_channels=img_channel, kernel_size=3,
                                                  padding=1, stride=1, groups=1, bias=False))
        self.ConvOut3_b = nn.Sequential(nn.PixelShuffle(2),  # 输入b*(2width)*(h/2)*(w/2) 输出b*(width/2)*h*w
                                        nn.Conv2d(in_channels=width // 2, out_channels=img_channel, kernel_size=3,
                                                  padding=1, stride=1, groups=1, bias=False))

        # fuse
        self.fuse1 = FFGM(chan, bias=bias)
        self.fuse2 = FFGM(chan * 2, bias=bias)
        self.off_up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.off_up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.wei_up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.wei_up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, inp):
        offsets = {}
        fin_res_sharps = []
        res_sharps_3 = []
        res3_head = []
        blur3 = inp

        # Encoder
        blur3 = self.patch_embed(blur3)
        blur3_out_enc1 = self.encoder_level1(blur3)  # 256,256,C
        blur3_inp_enc2 = self.sta3down1_2(blur3_out_enc1)  # 128,128,2C

        blur3_out_enc2 = self.encoder_level2(blur3_inp_enc2)  # 128,128,2C
        blur3_inp_enc3 = self.sta3down2_3(blur3_out_enc2)  # 64,64,4C

        blur3_out_enc3 = self.encoder_level3(blur3_inp_enc3)  # 64,64,4C

        # Middle
        offset = self.offset_conv1(blur3_out_enc3)
        inverse_offset = self.offset_conv2(offset)
        inverse_offset = self.offset_conv3(inverse_offset)  # 64,64,98

        weight = self.weight_conv1(blur3_out_enc3)
        weight = self.weight_activation1(weight)
        inverse_weight = self.weight_conv2(weight)
        inverse_weight = self.weight_conv3(inverse_weight)  # 64,64,49

        blur3_out_enc3 = self.inverse_conv(input=blur3_out_enc3, offset=inverse_offset, mask=inverse_weight)
        offsets.update(
            {'offset': offset, 'weight': weight, 'inverse_offset': inverse_offset, 'inverse_weight': inverse_weight})

        # Decoder
        blur3_out_dec3 = self.decoder_level3(blur3_out_enc3)  # 64,64,4C
        S3_a = self.ConvOut3_a(blur3_out_dec3)
        res_sharps_3.append(S3_a)

        blur3_inp_dec2 = self.sta3up3_2(blur3_out_dec3)  # 128,128,2C
        inverse_offset2 = self.off_up2(inverse_offset)  # 128,128,98
        inverse_weight2 = self.wei_up2(inverse_weight)  # 128,128,49

        blur3_inp_dec2 = self.fuse2(blur3_out_enc2, blur3_inp_dec2, inverse_offset2, inverse_weight2)
        blur3_out_dec2 = self.decoder_level2(blur3_inp_dec2)  # 128,128,128
        S3_b = self.ConvOut3_b(blur3_out_dec2)
        res_sharps_3.append(S3_b)

        blur3_inp_dec1 = self.sta3up2_1(blur3_out_dec2)  # 256,256,C
        inverse_offset1 = self.off_up1(inverse_offset2)  # 256,256,98
        inverse_weight1 = self.wei_up1(inverse_weight2)  # 256,256,49

        blur3_inp_dec1 = self.fuse1(blur3_out_enc1, blur3_inp_dec1, inverse_offset1, inverse_weight1)
        blur3_out_dec1 = self.decoder_level1(blur3_inp_dec1)  # 256,256,64
        S3_c = self.output(blur3_out_dec1)
        res_sharps_3.append(S3_c)

        for i in range(3):  # (0,0),(1,0),(1,1),(2,0),(2,1),(2,2)
            for j in range(i + 1):
                res3_head.append((res_sharps_3[i] + res_sharps_3[j]) / 2)
        res3_head.append((res_sharps_3[0] + res_sharps_3[1] + res_sharps_3[2]) / 3)
        res3_head = torch.stack(res3_head, dim=1)
        res3_head = res3_head + inp.unsqueeze(dim=1)

        fin_res_sharps.append(res3_head)
        return fin_res_sharps, offsets

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


class MBIG_DFN_Local(Local_Base, MBIG_DFN):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        MBIG_DFN.__init__(self, *args, **kwargs)
        # train_size = (1, 3, 64, 64)
        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))
        # base_size = (int(H), int(W))
        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)




