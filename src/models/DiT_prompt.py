# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from ..utils.download import find_model
from torch.nn import Conv2d, Dropout
from functools import reduce
from operator import mul
from .mlp import MLP
from .attention_fusion import AttentionFusion, ConvFusion
from einops import rearrange
import sys

from diffusers import StableDiffusion3Pipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from transformer_sd3_modified import SD3Transformer2DModel

import time

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

# class DiTBlock(nn.Module):
#     """
#     A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
#     """
#     def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
#         self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
#         self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
#         mlp_hidden_dim = int(hidden_size * mlp_ratio)
#         approx_gelu = lambda: nn.GELU(approximate="tanh")
#         self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
#         self.adaLN_modulation = nn.Sequential(
#             nn.SiLU(),
#             nn.Linear(hidden_size, 6 * hidden_size, bias=True)
#         )

#     def forward(self, x, c):
#         shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
#         #x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
#         x, attn_weights = self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
#         x = x + gate_msa.unsqueeze(1) * x
#         x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
#         return x, attn_weights


# class FinalLayer(nn.Module):
#     """
#     The final layer of DiT.
#     """
#     def __init__(self, hidden_size, patch_size, out_channels):
#         super().__init__()
#         self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
#         self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
#         self.adaLN_modulation = nn.Sequential(
#             nn.SiLU(),
#             nn.Linear(hidden_size, 2 * hidden_size, bias=True)
#         )

#     def forward(self, x, c):
#         shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
#         x = modulate(self.norm_final(x), shift, scale)
#         x = self.linear(x)
#         return x

# class DiT(nn.Module):
#     """
#     Diffusion model with a Transformer backbone.
#     """
#     def __init__(
#         self,
#         input_size=32,
#         patch_size=2,
#         in_channels=4,
#         hidden_size=1152,
#         depth=28,
#         num_heads=16,
#         mlp_ratio=4.0,
#         class_dropout_prob=0.1,
#         num_classes=1000,
#         learn_sigma=True,
#     ):
#         super().__init__()
#         self.learn_sigma = learn_sigma
#         self.in_channels = in_channels
#         self.out_channels = in_channels * 2 if learn_sigma else in_channels
#         self.patch_size = patch_size
#         self.num_heads = num_heads
#         self.hidden_size = hidden_size
#         self.depth = depth

#         self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
#         self.t_embedder = TimestepEmbedder(hidden_size)
#         self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
#         num_patches = self.x_embedder.num_patches
#         # Will use fixed sin-cos embedding:
#         self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

#         self.blocks = nn.ModuleList([
#             DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
#         ])
#         self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
#         self.initialize_weights()

#     def initialize_weights(self):
#         # Initialize transformer layers:
#         def _basic_init(module):
#             if isinstance(module, nn.Linear):
#                 torch.nn.init.xavier_uniform_(module.weight)
#                 if module.bias is not None:
#                     nn.init.constant_(module.bias, 0)
#         self.apply(_basic_init)

#         # Initialize (and freeze) pos_embed by sin-cos embedding:
#         pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
#         self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

#         # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
#         w = self.x_embedder.proj.weight.data
#         nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
#         nn.init.constant_(self.x_embedder.proj.bias, 0)

#         # Initialize label embedding table:
#         nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

#         # Initialize timestep embedding MLP:
#         nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
#         nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

#         # Zero-out adaLN modulation layers in DiT blocks:
#         for block in self.blocks:
#             nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
#             nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

#         # Zero-out output layers:
#         nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
#         nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
#         nn.init.constant_(self.final_layer.linear.weight, 0)
#         nn.init.constant_(self.final_layer.linear.bias, 0)

#     def unpatchify(self, x):
#         """
#         x: (N, T, patch_size**2 * C)
#         imgs: (N, H, W, C)
#         """
#         c = self.out_channels
#         p = self.x_embedder.patch_size[0]
#         h = w = int(x.shape[1] ** 0.5)
#         assert h * w == x.shape[1]

#         x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
#         x = torch.einsum('nhwpqc->nchpwq', x)
#         imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
#         return imgs

#     def forward(self, x, t, y):
#         """
#         Forward pass of DiT.
#         x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
#         t: (N,) tensor of diffusion timesteps
#         y: (N,) tensor of class labels
#         """
#         x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
#         t = self.t_embedder(t)                   # (N, D)
#         y = self.y_embedder(y, self.training)    # (N, D)
#         c = t + y                                # (N, D)     #here add prompt
#         for block in self.blocks:
#             #x,_ = block(x, c)
#             x = block(x, c)                       # (N, T, D)
#         x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
#         x = self.unpatchify(x)                   # (N, out_channels, H, W)
#         return x

#     def forward_with_cfg(self, x, t, y, cfg_scale):
#         """
#         Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
#         """
#         # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
#         half = x[: len(x) // 2]
#         combined = torch.cat([half, half], dim=0)
#         model_out = self.forward(combined, t, y)
#         # For exact reproducibility reasons, we apply classifier-free guidance on only
#         # three channels by default. The standard approach to cfg applies it to all channels.
#         # This can be done by uncommenting the following line and commenting-out the line following that.
#         # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
#         eps, rest = model_out[:, :3], model_out[:, 3:]
#         cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
#         half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
#         eps = torch.cat([half_eps, half_eps], dim=0)
#         return torch.cat([eps, rest], dim=1)



#     def load_pretrain(self, cfg):
#         ckpt_path = cfg.MODEL.CKPT or f"DiT-XL-2-{256}x{256}.pt"
#         state_dict = find_model(ckpt_path)
#         self.load_state_dict(state_dict)

class PromptedDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self, cfg, load_pretrain=True, vis=False,
    ):
        super().__init__()   
        #self.weight_dtype = torch.float16
        self.weight_dtype = torch.float16
        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            text_encoder_3=None,
            tokenizer_3=None,
            torch_dtype=torch.float16
        )
        self.pipe.transformer = SD3Transformer2DModel.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers", 
            subfolder="transformer", 
            revision=None, 
            variant=None,
            torch_dtype=torch.float16
        )

        self.freeze_model(self.pipe.transformer)
        self.encoder = self.pipe.transformer
        
        self.pipe.to("cuda")


        self.depth = cfg.MODEL.DEPTH
        self.cfg = cfg
        self.hidden_size = self.encoder.inner_dim
        
        

        #prompt
        self.prompt_config = cfg.MODEL.PROMPT
        num_tokens = self.prompt_config.NUM_TOKENS
        self.num_tokens = num_tokens
        self.prompt_dropout = Dropout(self.prompt_config.DROPOUT)

        # if project the prompt embeddings
        if self.prompt_config.PROJECT > -1:
            # only for prepend / add
            prompt_dim = self.prompt_config.PROJECT
            self.prompt_proj = nn.Linear(
                prompt_dim, self.hidden_size)      #config.hidden_size
            nn.init.kaiming_normal_(
                self.prompt_proj.weight, a=0, mode='fan_out')
        else:
            prompt_dim = self.hidden_size          #config.hidden_size
            self.prompt_proj = nn.Identity()

        #TODO how to decide patch_size??
        patch_size = (16,16)
        # initiate prompt:
        if self.prompt_config.INITIATION == "random":
            val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa
            if self.prompt_config.TYPE is not None:
                self.prompt_embeddings = nn.Parameter(torch.zeros(
                    1, num_tokens, prompt_dim))
                # xavier_uniform initialization
                nn.init.uniform_(self.prompt_embeddings.data, -val, val)

            if self.prompt_config.TYPE == "deep":        # noqa
                total_d_layer = self.depth - 1        #config.transformer["num_layers"]-1
                self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
                    total_d_layer, num_tokens, prompt_dim))
                # xavier_uniform initialization
                nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)
        else:
            raise ValueError("Other initiation scheme is not supported")

        if self.cfg.MODEL.FUSION_TYPE == "conv":
            self.merge_module = ConvFusion(cfg)
        elif self.cfg.MODEL.FUSION_TYPE == "linear":
            raise ValueError("Fusion type not supported")
        elif self.cfg.MODEL.FUSION_TYPE == "attention":
            self.merge_module = AttentionFusion(cfg)
        else:
            raise ValueError("Fusion type not supported")
        
        #add downstream head
        self.setup_head(cfg)

        #TODO: it should be removed  or it should be used to decrease output dimension
        # if len(cfg.MODEL.ATTENTION_LAYER_LIST) > 0:
        #     self.atten_proj = nn.Linear(4096, 1152)
        
        
    def setup_head(self, cfg):
        self.head = MLP(
            # input_dim= self.encoder.x_embedder.num_patches,
            # mlp_dims=[self.encoder.x_embedder.num_patches * 4] * self.cfg.MODEL.MLP_NUM + \
            #     [cfg.DATA.NUMBER_CLASSES], # noqa
            input_dim= self.merge_module.feature_dims,  
            mlp_dims=[self.merge_module.feature_dims * 4] * self.cfg.MODEL.MLP_NUM + \
                [cfg.DATA.NUMBER_CLASSES], # noqa
            special_bias=True
        )


    def train(self, mode=True):
        # set train status for this class: disable all but the prompt-related modules
        if mode:
            # training:
            self.encoder.eval()
            #self.embeddings.eval()
            self.prompt_proj.train()
            self.prompt_dropout.train()
            self.head.train()
            self.merge_module.train()
        else:
            # eval:
            for module in self.children():
                module.train(mode)

    def eval(self):
        self.encoder.eval()
        self.prompt_proj.eval()
        self.prompt_dropout.eval()
        self.head.eval()
        self.merge_module.eval()


    def load_pretrain(self):
        ckpt_path = self.cfg.MODEL.CKPT or f"DiT-XL-2-{256}x{256}.pt"
        state_dict = find_model(ckpt_path)
        self.load_state_dict(state_dict)

    def forward_deep_prompt(self, embedding_output, c):
        #attn_weights = []
        features = []
        hidden_states = None
        #weights = None
        B = embedding_output.shape[0]
        num_layers = self.depth

        for i in range(self.depth):
            if i == 0:
                #hidden_states, weights = self.blocks[i](embedding_output)
                hidden_states, attn_weights = self.encoder.blocks[i](embedding_output, c)
                
            else:
                if i <= self.deep_prompt_embeddings.shape[0]:
                    deep_prompt_emb = self.prompt_dropout(self.prompt_proj(
                        self.deep_prompt_embeddings[i-1]).expand(B, -1, -1))

                    hidden_states = torch.cat((
                        deep_prompt_emb,
                        hidden_states[:, self.num_tokens:, :]
                    ), dim=1)

                #hidden_states, weights =  self.blocks[i](hidden_states)
                hidden_states, attn_weights =  self.encoder.blocks[i](hidden_states, c)
            
            if i in self.cfg.MODEL.FEATURES_LAYER_LIST:
                features.append(hidden_states[:, self.num_tokens:, :])
            if i in self.cfg.MODEL.ATTENTION_LAYER_LIST:
                attn_weights = rearrange(attn_weights[:,:,self.num_tokens:, self.num_tokens:], 'b heads length dim -> b length (heads dim)')
                features.append(self.atten_proj(attn_weights))

            # if self.encoder.vis:
            #     attn_weights.append(weights)

        return features   # hidden_states, attn_weights

    def forward_shallow_prompt(self, embedding_output, c):
        #attn_weights = []
        features = []
        hidden_states = None
        #weights = None
        B = embedding_output.shape[0]


        for i in range(self.depth):
            hidden_states, attn_weights = self.encoder.blocks[i](embedding_output, c)
            if i in self.cfg.MODEL.FEATURES_LAYER_LIST:
                features.append(hidden_states[:, self.num_tokens:, :])
            if i in self.cfg.MODEL.ATTENTION_LAYER_LIST:
                attn_weights = rearrange(attn_weights[:,:,self.num_tokens:, self.num_tokens:], 'b heads length dim -> b length (heads dim)')
                features.append(self.atten_proj(attn_weights))
        return features
    
    def forward_no_prompt(self, hidden_states, encoder_hidden_states, temb):
        return_list_dict = {}
        for index_block, block in enumerate(self.encoder.transformer_blocks):
            encoder_hidden_states, hidden_states, feat_dict = block(
                 hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, temb=temb
                )
            feat_dict['feats'] = hidden_states
            if index_block > max(self.cfg.MODEL.FEATURES_LAYER_LIST):
                break
            if index_block in self.cfg.MODEL.FEATURES_LAYER_LIST:
                for key, value in feat_dict.items():
                    if key in self.cfg.MODEL.FEATURES_TYPE_LIST:
                        if key not in return_list_dict:
                            return_list_dict[key] = []
                        #return_list_dict[key].append(value.cpu())
                        return_list_dict[key].append(value)

        
        return return_list_dict

    def downstream_head(self, x):
        x = self.head(x)
        return x

    def freeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def merge_features_old(self, features):
        features = torch.cat(features, dim=1).permute(0, 2, 1)
        features = self.merge_module(features)
        features = features.squeeze(-1)
        return  features

    def merge_features_old_2(self, features):
        features = torch.cat(features, dim=-1)
        features = self.merge_module(features)
        features = features.squeeze(-1)
        return  features
    
    def merge_features(self, features):
        features = self.merge_module(features)
        return  features



    def forward(self, 
                hidden_states_list,
                timesteps_list, 
                encoder_hidden_states, 
                pooled_projections
                ):
        # B = x_list[0].shape[0]
        # y = torch.tensor([1000] * B).to(x_list[0].device)       #label =1000, do unconditional sampling
        # y = self.encoder.y_embedder(y, self.encoder.training)                   # (N, D)

        # features_list = []
        # for idx, t in enumerate(self.cfg.MODEL.T_LIST):
        #     t = torch.tensor([t] * B).to(x_list[0].device)       
        #     #x_t = self.add_noise(x, t)    
        #     x_t = x_list[idx]
        #     x_t = self.encoder.x_embedder(x_t) + self.encoder.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        #     t = self.encoder.t_embedder(t)                   # (N, D)
        #     c = t + y                                # (N, D)     
        features_list = []
        encoder_hidden_states_copy = encoder_hidden_states
        for idx, timestep in enumerate(timesteps_list):
            hidden_states = self.encoder.pos_embed(hidden_states_list[idx]) 
            temb = self.encoder.time_text_embed(timestep, pooled_projections)
            encoder_hidden_states = self.encoder.context_embedder(encoder_hidden_states_copy)

            # incorporate prompt
            if self.prompt_config.TYPE is not None:
                hidden_states = torch.cat((                                   #remove cls_ token
                    self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(hidden_states.shape[0], -1, -1)),
                    hidden_states
                        ), dim=1)


            features_t = []
            if self.prompt_config.TYPE == "deep":
                features_t = self.forward_deep_prompt(hidden_states, encoder_hidden_states, temb)
            elif self.prompt_config.TYPE == "shallow":
                features_t = self.forward_shallow_prompt(hidden_states, encoder_hidden_states, temb)
            else:
                features_t = self.forward_no_prompt(hidden_states, encoder_hidden_states, temb)

            features_list.append(features_t)
        

                
        #print(f"t1: {time.time()}")
        if self.cfg.MODEL.SAVE_FEATURES:
            #just save features from the first batch 
            self.save_featues(features_list, timesteps_list) 
            sys.exit()           
        #print(f"t2: {time.time()}")
        features_list = self.flatten_list(features_list)
        features = self.merge_features(features_list)
        result = self.head(features)
        #print(f"t3: {time.time()}")
        return result

    def save_featues(self, features_list, timesteps_list,  path="./features"):
        for idx, timestep in enumerate(timesteps_list):
            features_dict = features_list[idx]
            for key, features in features_dict.items():
                features = torch.cat(features, dim=0)
                features = features.cpu().detach().numpy()
                np.save("/home/local/ASUAD/changyu2/prompt_DiT/features/base_feature_SD3/"+ f"t_{int(timestep.item())}_{key}.npy", features)
    
    def flatten_list(self, features_list):
        features_list_flatten = []
        for feature_t in features_list:
            for key, features in feature_t.items():
                features_list_flatten.extend(features)
        features_list_flatten = [feature.to(torch.float32) for feature in features_list_flatten]
        return features_list_flatten
#overwrite ATTENTION, just change the forward function, return attn weights
class Attention(Attention):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        #remove scaled_dot_product_attention
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        #return x, attn
        return x, {'map': attn, 
                  'query': rearrange(q, 'b heads length dim -> b length (heads dim)'), 
                   'key': rearrange(k, 'b heads length dim -> b length (heads dim)'), 
                   'value': rearrange(v, 'b heads length dim -> b length (heads dim)')}



#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}
