import torch
import torch.nn as nn
import torchvision.models as models
from einops import rearrange, repeat

class blip_resnet(nn.Module):
    def __init__(self, args):
        super(blip_resnet, self).__init__()
        model = getattr(models, 'resnet50')(pretrained=True)
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)
        map_size = int(args.image_size / 32)
        self.avg_fnt = torch.nn.AvgPool2d(kernel_size=map_size, stride=1, padding=0)
    def forward(self, x):
        if len(x.shape) == 5:
            images = x
            b,t,c,h,w = images.shape
            images = rearrange(images,'b t c h w -> (b t) c h w')
            batch_size = images.shape[0]
            feat_size = 2048 # [5,25,3,224,224]
            patch_feats = self.model(images).reshape(batch_size, feat_size, -1).permute(0, 2, 1)
            patch_feats = rearrange(patch_feats,'(b t) v d -> b (t v) d',b=b,t=t)
            avg_feats = torch.mean(patch_feats, dim=1)
            return patch_feats, avg_feats
        elif x.dim() == 6:
            b, num_seq, t, c, h, w = x.shape
            x = rearrange(x,'b num_seq t c h w -> (b num_seq t) c h w')
            patch_feats = self.model(x)
            bs, feat_dim = patch_feats.shape[:2]
            patch_feats = patch_feats.reshape(bs, feat_dim, -1).permute(0, 2, 1)
            patch_feats = rearrange(patch_feats,'(b num_seq t) v d -> (b num_seq) (t v) d', b=b, num_seq=num_seq, t=t)
            return patch_feats
        else:     
            patch_feats = self.model(x)
            avg_feats = self.avg_fnt(patch_feats).flatten(1)
            batch_size, feat_size, _, _ = patch_feats.shape
            # NxLxD
            patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
            return patch_feats, avg_feats

