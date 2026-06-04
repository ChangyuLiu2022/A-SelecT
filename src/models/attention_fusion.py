import torch
import torch.nn as nn
from .attention_head import AttentionHead, LambdaLayer
from einops import rearrange 

def concate_dict_feats(features_dict):
        feat_list = []
        for key, value in features_dict.items():
            feat_list = feat_list + value
        feat_block = torch.cat(feat_list, dim=1)
        return feat_block


class AttentionFusion(nn.Module):
    def __init__(self, args):
        super(AttentionFusion, self).__init__()
        self.cfg = args
        attention_dims = int(self.cfg.MODEL.FUSION_ARC.split(',')[0].strip().split(':')[2])
        self.feature_num = len(self.cfg.MODEL.T_LIST) * len(self.cfg.MODEL.FEATURES_LAYER_LIST) * len(self.cfg.MODEL.FEATURES_TYPE_LIST)
        self.feature_dims = attention_dims * self.feature_num 
        feat_size = int(self.cfg.DATA.CROPSIZE/16)    #h and w
        t_len = len(self.cfg.MODEL.T_LIST)
        type_len = len(self.cfg.MODEL.FEATURES_TYPE_LIST)
        feat_len = len(self.cfg.MODEL.FEATURES_LAYER_LIST)
        self.block_attention_list = nn.ModuleList()
        for i in range(self.feature_num):
            block_attention = AttentionHead(self.cfg.MODEL.FUSION_ARC.split("/")[0])
            self.block_attention_list.append(block_attention)
        
        
        if self.cfg.MODEL.FUSION_PRE_LAYER:
            self.pre_layer_list = nn.ModuleList()
            if self.cfg.MODEL.FUSION_PRE_LAYER_TYPE == 'attn_reso_dim':
                #change both of attention and resolution dimension
                for i in range(self.feature_num):
                    conv =  nn.Conv2d(1536 * feat_len * type_len, attention_dims, 3, 2, 1)
                    pre_layer = nn.Sequential(
                        LambdaLayer(lambda x: rearrange(x, 'b (l h w) c -> b (l c) h w', h = feat_size, w = feat_size, l = feat_len * type_len)),
                        conv,
                        LambdaLayer(lambda x: rearrange(x, 'b c h w -> b (h w) c')),
                    )
                    self.pre_layer_list.append(pre_layer)
                
            elif self.cfg.MODEL.FUSION_PRE_LAYER_TYPE == 'attn_dim':
                #change only attention dimension
                for i in range(self.feature_num):
                    pre_layer = nn.Linear(1536, attention_dims)
                    self.pre_layer_list.append(pre_layer)
            elif self.cfg.MODEL.FUSION_PRE_LAYER_TYPE == 'reso_dim':
                #change only resolution dimension
                liner_input_dim = feat_size * feat_size  * feat_len * type_len
                liner_out_dim = self.cfg.MODEL.FUSION_LINEAR_OUT_DIM
                for i in range(self.feature_num):
                    tran_reso_dim = nn.Linear(liner_input_dim, liner_out_dim)
                    pre_layer = nn.Sequential(
                        LambdaLayer(lambda x: rearrange(x, 'b c d -> b d c')),
                        tran_reso_dim,
                        LambdaLayer(lambda x: rearrange(x, 'b d c -> b c d')),
                    )
                    self.pre_layer_list.append(pre_layer)
            
    def forward(self, features_list):
        inter_noise_step_feat = []
        for i in range(self.feature_num):            
            if self.cfg.MODEL.FUSION_PRE_LAYER:
                x = self.pre_layer_list[i](features_list[i])
            else:
                x = features_list[i]
            x = self.block_attention_list[i](x)
            inter_noise_step_feat.append(x)
        x = torch.concat(inter_noise_step_feat, dim=1)
        return x


class ConvFusion(nn.Module):
    def __init__(self, args):
        super(ConvFusion, self).__init__()
        self.cfg = args
        type_len = len(self.cfg.MODEL.FEATURES_TYPE_LIST)
        feat_len = len(self.cfg.MODEL.FEATURES_LAYER_LIST)
        input_dim = 1536 * feat_len * type_len
        intermed_dim = 1536 if input_dim/2 <1536 else int(input_dim/2)
        
        self.feat_size = int(self.cfg.DATA.CROPSIZE/16)
        kernel_size = 2
        stride_size = 2
        #initialize a two layer conv network
        self.conv1 = nn.Conv2d(input_dim, intermed_dim, kernel_size, stride_size)
        self.conv2 = nn.Conv2d(intermed_dim, 200, kernel_size, stride_size)
        self.relu = nn.ReLU()
        self.feature_dims = 200 * (int(self.feat_size/(kernel_size * kernel_size)))**2 * len(self.cfg.MODEL.T_LIST)
        #square of the feature size divided by the kernel size squared

    def forward(self, features_list):
        inter_noise_step_feat = []
        for i in range(len(self.cfg.MODEL.T_LIST)):
            x = concate_dict_feats(features_list[i])
            x = rearrange(x, 'b (h w) c -> b c h w', h = self.feat_size, w = self.feat_size)
            x = x.to(torch.float32)
            x = self.conv1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.relu(x)
            #x = rearrange(x, 'b c h w -> b (h w) c')
            x = x.reshape(x.size(0), -1)
            inter_noise_step_feat.append(x)
        x = torch.concat(inter_noise_step_feat, dim=1)
        return x
    
#write a 3 layer conv network to classify images


