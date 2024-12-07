from pyexpat import features

import torch.nn as nn
import torch
from clip import model
from clip import clip
from models.vit import *
import numpy as np
from clip.model import ResidualAttentionBlock
from config import argument_parser
parser = argument_parser()
args = parser.parse_args()
def trans_data(x, y, alpha=1.0, natt=1, c_in=768):
    batch_size = x.shape[0]

    x_new = torch.zeros(x.shape).cuda()
    y_new = torch.zeros(y.shape).cuda()
    for i in range(natt):
        index = torch.randperm(batch_size).cuda()
        x_new[:, i, :] = x[index][:, i, :]
        y_new[:, i] = y[index][:, i]

    lam = np.random.beta(alpha, alpha, natt)
    lam = torch.from_numpy(lam).cuda().float()
    lam = torch.reshape(lam, [1, natt, 1])

    lam_n = np.random.beta(alpha, alpha, [batch_size, natt])
    lam_n = torch.from_numpy(lam_n).cuda().float()
    lam_n = torch.reshape(lam_n, [batch_size, natt, 1])

    lam_v = np.random.beta(alpha, alpha, [batch_size, natt])
    lam_v = torch.from_numpy(lam_v).cuda().float()
    lam_v = torch.reshape(lam_v, [batch_size, natt, 1])

    norm = torch.reshape(torch.norm(x, p=2, dim=2), [batch_size, natt, 1])
    vec = nn.functional.normalize(x, p=2, dim=2, eps=1e-12)

    norm_new = torch.reshape(torch.norm(x_new, p=2, dim=2), [batch_size, natt, 1])
    vec_new = nn.functional.normalize(x_new, p=2, dim=2, eps=1e-12)

    vec = vec * lam_v + vec_new * (1 - lam_v)
    norm = norm * lam_n + norm_new * (1 - lam_n)
    x_m = vec * norm

    eq_index = y == y_new
    #eq_index = repeat(eq_index, 'b a -> b a c', c=c_in)
    eq_index = eq_index.unsqueeze(-1)  # (4, 26) -> (4, 26, 1)
    eq_index = eq_index.expand(-1, -1, c_in)  # (4, 26, 1) -> (4, 26, 2048)
    x_u = lam * x + (1 - lam) * x_new
    mixed_x = torch.where(eq_index, x_m, x_u)

    lam = torch.reshape(lam, [1, natt])
    y = lam * y + (1 - lam) * y_new

    return mixed_x, y
class TransformerClassifier(nn.Module):
    def __init__(self, clip_model, attr_num, attributes, dim=768, pretrain_path='/media/a1/E10/RobustPAR/data/pretrain_model/jx_vit_base_p16_224-80ecf9dd.pth'):
        super().__init__()
        super().__init__()
        self.attr_num = attr_num
        self.word_embed = nn.Linear(clip_model.visual.output_dim, dim)
        vit = vit_base()
        vit1 = vit_base()
        vit2 = vit_base()
        vit.load_param(pretrain_path)
        vit1.load_param(pretrain_path)
        vit2.load_param(pretrain_path)
        self.vit1 = vit1.blocks1[-1:]
        self.vit2 = vit2.blocks2[-1:]
        self.norm00 = vit.norm
        self.norm11 = vit.norm1
        self.norm22 = vit.norm2
        self.weight_layer0 = nn.ModuleList([nn.Linear(dim, 1) for i in range(self.attr_num)])
        self.weight_layer1 = nn.ModuleList([nn.Linear(dim, 1) for i in range(self.attr_num)])
        self.dim = dim #768
        self.text = clip.tokenize(attributes).to("cuda")
        self.bn0 = nn.BatchNorm1d(self.attr_num)
        self.bn1 = nn.BatchNorm1d(self.attr_num)
        self.attn = nn.MultiheadAttention(dim, 12)
        fusion_len = self.attr_num + 257 + args.vis_prompt
        if not args.use_mm_former :
            print('Without MM-former, Using MLP Instead')
            self.linear_layer = nn.Linear(fusion_len, self.attr_num)
        else:
            self.blocks0 = vit.blocks[-args.mm_layers:]
    def forward(self,imgs,clip_model,gt_label,mode = 'train'):
        b_s=imgs.shape[0]
        clip_image_features,all_class,attenmap=clip_model.visual(imgs.type(clip_model.dtype))
        text_features = clip_model.encode_text(self.text).to("cuda").float()
        if args.use_div:
            final_similarity, logits_per_image = clip_model.forward_aggregate(all_class, text_features)
        else :
            final_similarity = None
        textual_features = self.word_embed(text_features).expand(b_s, self.attr_num, self.dim)

        clip_image_features1 = clip_image_features.permute(1, 0, 2).float()
        textual_features1 = textual_features.permute(1, 0, 2).float()
        text_video_video , _= self.attn(textual_features1, clip_image_features1, clip_image_features1,need_weights=True) #
        text_video_video = text_video_video.permute(1, 0, 2)

        for blk in self.vit1:
           x_vit1= blk(text_video_video)
        x_vit1 =self.norm11(x_vit1)
        for blk in self.vit2:
           x_vit2=blk(x_vit1)
        x_vit2=self.norm22(x_vit2+x_vit1)

        x = torch.cat([textual_features, clip_image_features], dim=1)
        if args.use_mm_former:
            for blk in self.blocks0:
                x = blk(x)
        else :
            x = x.permute(0, 2, 1)
            x= self.linear_layer(x)
            x = x.permute(0, 2, 1)
        x = self.norm00(x)

        x_split1 = x[:, :self.attr_num, :]
        if mode == 'train':
            x_split1, gt_label1 = trans_data(x=x_split1, y=gt_label, natt=self.attr_num, c_in=self.dim)
            x_vit2, gt_label2 = trans_data(x=x_vit2, y=gt_label, natt=self.attr_num, c_in=self.dim)
        else:
            gt_label1 = gt_label
            gt_label2 = gt_label

        logits = torch.cat([self.weight_layer0[i](x_split1[:, i, :]) for i in range(self.attr_num)], dim=1)
        bn_logits = self.bn0(logits)
        logits_cross =torch.cat([self.weight_layer1[i](x_vit2[:, i, :]) for i in range(self.attr_num)], dim=1)
        bn_logits2 = self.bn1(logits_cross)

        return bn_logits,final_similarity,bn_logits2,gt_label1,gt_label2