import random
from typing import List

import torch
from torch import nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from torch.autograd import Function
from ops.occ_pooling import occ_pool
from layers.backbones.cfs import CFS3d
from layers.backbones.cbam3d_v2 import CBAM3d_v2
from axial_attention import AxialAttention

def conv3x3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False,
                     dilation=dilation)

def get_norm_3d(norm, out_channels):
    """ Get a normalization module for 3D tensors
    Args:
        norm: (str or callable)
        out_channels
    Returns:
        nn.Module or None: the normalization layer
    """

    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "BN": nn.BatchNorm3d,
            "GN": lambda channels: nn.GroupNorm(32, channels),
            "nnSyncBN": nn.SyncBatchNorm,  # keep for debugging
        }[norm]
    return norm(out_channels)

class BasicBlock3d(nn.Module):
    """ 3x3x3 Resnet Basic Block"""
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm='BN', drop=0):
        super(BasicBlock3d, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3x3(inplanes, planes, stride, 1, dilation)
        self.bn1 = get_norm_3d(norm, planes)
        self.drop1 = nn.Dropout(drop, True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, 1, 1, dilation)
        self.bn2 = get_norm_3d(norm, planes)
        self.drop2 = nn.Dropout(drop, True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.drop1(out) # drop after both??
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.drop2(out) # drop after both??

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class volume_reduce(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(volume_reduce, self).__init__()
        self.model = nn.Sequential(
            BasicBlock3d(in_channels, in_channels),
            self._get_conv(in_channels, in_channels * 2),
            BasicBlock3d(in_channels * 2, in_channels * 2),
            self._get_conv(in_channels * 2, in_channels * 4),
            BasicBlock3d(in_channels * 4, in_channels * 4),
            self._get_conv(in_channels * 4, out_channels)
        )

    def _get_conv(self, in_channels, out_channels, stride=(1, 1, 2), padding=(1, 1, 1)):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.model(x)
        #print('debug4:', x.size())
        assert x.shape[-1] == 1
        return x[..., 0].transpose(-1, -2)
        #return x[..., 0]

class VisFuser(nn.Module):
    def __init__(self, in_channels, out_channels, norm_cfg=None) -> None:
        super(VisFuser, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if norm_cfg is None:
            norm_cfg = dict(type='BN3d', eps=1e-3, momentum=0.01)

        self.img_enc = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 7, padding=3, bias=False),
            build_norm_layer(norm_cfg, out_channels)[1],
            # nn.BatchNorm3d(out_channels),
            nn.ReLU(True),
        )
        self.pts_enc = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 7, padding=3, bias=False),
            build_norm_layer(norm_cfg, out_channels)[1],
            # nn.BatchNorm3d(out_channels),
            nn.ReLU(True),
        )
        self.vis_enc = nn.Sequential(
            nn.Conv3d(2*out_channels, 16, 3, padding=1, bias=False),
            build_norm_layer(norm_cfg, 16)[1],
            # nn.BatchNorm3d(16),
            nn.ReLU(True),
            nn.Conv3d(16, 1, 1, padding=0, bias=False),
            nn.Sigmoid(),
        )

        self.volume_pp = volume_reduce(out_channels, out_channels)

    def forward(self, img_voxel_feats, pts_voxel_feats):

        img_voxel_feats = self.img_enc(img_voxel_feats)
        pts_voxel_feats = self.pts_enc(pts_voxel_feats)
        vis_weight = self.vis_enc(torch.cat([img_voxel_feats, pts_voxel_feats], dim=1))
        voxel_feats = vis_weight * img_voxel_feats + (1 - vis_weight) * pts_voxel_feats
        #print('debug3:', voxel_feats.size())
        voxel_feats2d = self.volume_pp(voxel_feats)
        #print('debug5:', voxel_feats2d.size())
        return voxel_feats2d

class AddFuser(nn.Module):
    def __init__(self, in_channels, out_channels, norm_cfg=None) -> None:
        super(AddFuser, self).__init__()    
        self.volume_pp = volume_reduce(out_channels, out_channels)

    def forward(self, img_voxel_feats, pts_voxel_feats):
        voxel_feats = img_voxel_feats + pts_voxel_feats
        #print('debug3:', voxel_feats.size())
        voxel_feats2d = self.volume_pp(voxel_feats)
        #print('debug5:', voxel_feats2d.size())
        return voxel_feats2d

class volume_reduce_light(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(volume_reduce_light, self).__init__()
        self.model = nn.Sequential(
            BasicBlock3d(in_channels, in_channels),
            self._get_conv(in_channels, in_channels),
            BasicBlock3d(in_channels, in_channels),
            self._get_conv(in_channels, in_channels * 2),
            BasicBlock3d(in_channels * 2, in_channels * 2),
            self._get_conv(in_channels * 2, out_channels)
        )

    def _get_conv(self, in_channels, out_channels, stride=(1, 1, 2), padding=(1, 1, 1)):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.model(x)
        #print('debug4:', x.size())
        assert x.shape[-1] == 1
        return x[..., 0].transpose(-1, -2)
        #return x[..., 0]

class CatFuser(nn.Module):
    def __init__(self, in_channels, out_channels, norm_cfg=None) -> None:
        super(CatFuser, self).__init__()
        self.volume_pp = volume_reduce_light(2*out_channels, out_channels)

    def forward(self, img_voxel_feats, pts_voxel_feats):
        voxel_feats = torch.cat((img_voxel_feats, pts_voxel_feats), 1)
        #print('debug3:', voxel_feats.size())
        voxel_feats2d = self.volume_pp(voxel_feats)
        #print('debug5:', voxel_feats2d.size())
        return voxel_feats2d

class ChannelFusion(nn.Module):
    def __init__(self, in_channels, out_channels, norm_cfg=None) -> None:
        super(ChannelFusion, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if norm_cfg is None:
            norm_cfg = dict(type='BN3d', eps=1e-3, momentum=0.01)

        self.img_enc = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 7, padding=3, bias=False),
            build_norm_layer(norm_cfg, out_channels)[1],
            # nn.BatchNorm3d(out_channels),
            nn.ReLU(True),
        )
        self.pts_enc = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 7, padding=3, bias=False),
            build_norm_layer(norm_cfg, out_channels)[1],
            # nn.BatchNorm3d(out_channels),
            nn.ReLU(True),
        )
        
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(2*out_channels, 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(16, out_channels, bias=False),
            nn.Sigmoid()
        )
        '''
        self.vis_enc = nn.Sequential(
            nn.Conv3d(2*out_channels, 16, 3, padding=1, bias=False),
            build_norm_layer(norm_cfg, 16)[1],
            # nn.BatchNorm3d(16),
            nn.ReLU(True),
            nn.Conv3d(16, 1, 1, padding=0, bias=False),
            nn.Sigmoid(),
        )
        '''
        self.volume_pp = volume_reduce(out_channels, out_channels)

    def forward(self, img_voxel_feats, pts_voxel_feats):

        img_voxel_feats = self.img_enc(img_voxel_feats)
        pts_voxel_feats = self.pts_enc(pts_voxel_feats)
        cat_feats = torch.cat([img_voxel_feats, pts_voxel_feats], dim=1)
        b, c, _, _, _ = img_voxel_feats.size()
        w0 = self.avg_pool(cat_feats).view(b, 2*c)
        w1 = self.fc(w0).view(b, c, 1, 1, 1)
        voxel_feats = w1 * img_voxel_feats + (1 - w1) * pts_voxel_feats
        #print('debug3:', voxel_feats.size())
        voxel_feats2d = self.volume_pp(voxel_feats)
        #print('debug5:', voxel_feats2d.size())
        return voxel_feats2d

class CFSFusion(nn.Module):
    def __init__(self, in_channels, out_channels, norm_cfg=None) -> None:
        super(CFSFusion, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if norm_cfg is None:
            norm_cfg = dict(type='BN3d', eps=1e-3, momentum=0.01)

        self.img_enc = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 7, padding=3, bias=False),
            build_norm_layer(norm_cfg, out_channels)[1],
            # nn.BatchNorm3d(out_channels),
            nn.ReLU(True),
        )
        self.pts_enc = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 7, padding=3, bias=False),
            build_norm_layer(norm_cfg, out_channels)[1],
            # nn.BatchNorm3d(out_channels),
            nn.ReLU(True),
        )

        self.cbam3d_fusion = CFS3d(2*in_channels)
        self.volume_pp = volume_reduce(out_channels, out_channels)

    def forward(self, img_voxel_feats, pts_voxel_feats):

        img_voxel_feats = self.img_enc(img_voxel_feats)
        pts_voxel_feats = self.pts_enc(pts_voxel_feats)
        voxel_feats = self.cbam3d_fusion(img_voxel_feats, pts_voxel_feats)
        #print('debug3:', voxel_feats.size())
        voxel_feats2d = self.volume_pp(voxel_feats)
        #print('debug5:', voxel_feats2d.size())
        return voxel_feats2d


class CBAMFusionV2(nn.Module):
    def __init__(self, in_channels, out_channels, norm_cfg=None) -> None:
        super(CBAMFusionV2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if norm_cfg is None:
            norm_cfg = dict(type='BN3d', eps=1e-3, momentum=0.01)

        self.img_enc = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 7, padding=3, bias=False),
            build_norm_layer(norm_cfg, out_channels)[1],
            # nn.BatchNorm3d(out_channels),
            nn.ReLU(True),
        )
        self.pts_enc = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 7, padding=3, bias=False),
            build_norm_layer(norm_cfg, out_channels)[1],
            # nn.BatchNorm3d(out_channels),
            nn.ReLU(True),
        )

        self.cbam3d_fusion = CBAM3d_v2(2*in_channels)
        self.volume_pp = volume_reduce(out_channels, out_channels)

    def forward(self, img_voxel_feats, pts_voxel_feats):

        img_voxel_feats = self.img_enc(img_voxel_feats)
        pts_voxel_feats = self.pts_enc(pts_voxel_feats)
        voxel_feats = self.cbam3d_fusion(img_voxel_feats, pts_voxel_feats)
        #print('debug3:', voxel_feats.size())
        voxel_feats2d = self.volume_pp(voxel_feats)
        #print('debug5:', voxel_feats2d.size())
        return voxel_feats2d


class AxialAttFusion(nn.Module):
    def __init__(self, in_channels, out_channels, norm_cfg=None) -> None:
        super(AxialAttFusion, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels//2
        if norm_cfg is None:
            norm_cfg = dict(type='BN3d', eps=1e-3, momentum=0.01)

        self.img_enc = nn.Sequential(
            nn.Conv3d(in_channels, self.out_channels, 7, padding=3, bias=False),
            build_norm_layer(norm_cfg, self.out_channels)[1],
            # nn.BatchNorm3d(out_channels),
            nn.ReLU(True),
        )
        self.pts_enc = nn.Sequential(
            nn.Conv3d(in_channels, self.out_channels, 7, padding=3, bias=False),
            build_norm_layer(norm_cfg, self.out_channels)[1],
            # nn.BatchNorm3d(out_channels),
            nn.ReLU(True),
        )

        self.axial_fusion = AxialAttention(
                                    dim = 256,           # embedding dimension
                                    dim_index = 2,       # where is the embedding dimension
                                    heads = 8,           # number of heads for multi-head attention
                                    num_dimensions = 3,  # number of axial dimensions (images is 2, video is 3, or more)
                                )
        self.volume_pp = volume_reduce(out_channels, out_channels)

    def forward(self, img_voxel_feats, pts_voxel_feats):

        #img_voxel_feats = self.img_enc(img_voxel_feats)
        #pts_voxel_feats = self.pts_enc(pts_voxel_feats)
        cat_feats = torch.cat([img_voxel_feats, pts_voxel_feats], dim=1)  # Bx80x256x256x8
        voxel_feats = self.axial_fusion(cat_feats)
        #print('debug3:', voxel_feats.size())
        voxel_feats2d = self.volume_pp(voxel_feats)
        #print('debug5:', voxel_feats2d.size())
        return voxel_feats2d

def openocc_voxel_pooling(geom_feats, x, dx, bx, nx):
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W
        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (bx - dx / 2.)) / dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < nx[0]) \
               & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < nx[1]) \
               & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]
        
        # [b, c, z, x, y] == [b, c, x, y, z]
        final = occ_pool(x, geom_feats, B, nx[2], nx[0], nx[1])  # ZXY
        final = final.permute(0, 1, 3, 4, 2)  # XYZ

        return final
