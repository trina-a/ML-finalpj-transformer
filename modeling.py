# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ml_collections

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage



#1.构建Embedding模块
class Embeddings(nn.Module):
    '''
    对图像进行编码，把图片当做一个句子，把图片分割成块，每一块表示一个单词
    '''
    def __init__(self,config,img_size,in_channels=3):
        super(Embeddings,self).__init__()
        img_size=img_size
        patch_size=config.patches["size"]
        n_patches=(img_size//patch_size)*(img_size//patch_size)

        self.patch_embeddings=Conv2d(in_channels=in_channels,
                                     out_channels=config.hidden_size,
                                     kernel_size=patch_size,
                                     stride=patch_size)
        self.position_embeddings=nn.Parameter(torch.zeros(1,n_patches+1,config.hidden_size))
        self.classifer_token=nn.Parameter(torch.zeros(1,1,config.hidden_size))
        self.dropout=Dropout((config.transformer["dropout_rate"]))



    def forward(self,x):
        bs=x.shape[0]
        cls_tokens=self.classifer_token.expand(bs,-1,-1)
        x=self.patch_embeddings(x)
        x=x.flatten(2)
        x=x.transpose(-1,-2)
        x=torch.cat((cls_tokens,x),dim=1)
        embeddings=x+self.position_embeddings
        embeddings=self.dropout(embeddings)
        return  embeddings



# def get_config():
#     """Returns the ViT-B/16 configuration."""
#     config = ml_collections.ConfigDict()
#     config.patches = ml_collections.ConfigDict({'size':2})
#     config.hidden_size = 500
#     config.transformer = ml_collections.ConfigDict()
#     config.transformer.mlp_dim = 2000
#     config.transformer.num_heads = 12
#     config.transformer.num_layers = 12
#     config.transformer.attention_dropout_rate = 0.0
#     config.transformer.dropout_rate = 0.1
#     config.classifier = 'token'
#     config.representation_size = None
#     return config
# config=get_config()



# ############################测试1.Embedding模块########################################
# embedding=Embeddings(config,img_size=32)
# #模拟图片信息
# img=torch.rand(4,3,32,32)
# out_embedding=embedding(img)
# print("1:img_embedding shape:",out_embedding.shape)
# #####################################################################################



#2.构建self-Attention模块
class Attention(nn.Module):
    def __init__(self,config,vis):
        super(Attention,self).__init__()
        self.vis=vis
        self.num_attention_heads=config.transformer["num_heads"]#12
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)  # 504/12=42
        self.all_head_size = self.num_attention_heads * self.attention_head_size  # 12*42=504

        self.query = Linear(config.hidden_size, self.all_head_size)#wm,504->504，Wq矩阵为（504,504）
        self.key = Linear(config.hidden_size, self.all_head_size)#wm,504->504,Wk矩阵为（504,504）
        self.value = Linear(config.hidden_size, self.all_head_size)#wm,504->504,Wv矩阵为（504,504）
        self.out = Linear(config.hidden_size, config.hidden_size)  # wm,504->504
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
        self.num_attention_heads, self.attention_head_size)  # wm,(bs,197)+(12,42)=(bs,197,12,42)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # wm,(bs,12,197,42)

    def forward(self, hidden_states):
        # hidden_states为：(bs,197,504)
        mixed_query_layer = self.query(hidden_states)#wm,504->504
        mixed_key_layer = self.key(hidden_states)#wm,504->504
        mixed_value_layer = self.value(hidden_states)#wm,504->504

        query_layer = self.transpose_for_scores(mixed_query_layer)#wm，(bs,12,197,42)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))#将q向量和k向量进行相乘（bs,12,197,197)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)#将结果除以向量维数的开方
        attention_probs = self.softmax(attention_scores)#将得到的分数进行softmax,得到概率
        weights = attention_probs if self.vis else None#wm,实际上就是权重
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)#将概率与内容向量相乘
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)#wm,(bs,197)+(504,)=(bs,197,504)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights#wm,(bs,197,504),(bs,197,197)


# ################测试2.self-Attention###############################
# attention=Attention(config,vis=True)
# out_selfattention,_=attention(out_embedding)
# print("2:out_selfattention shape:",out_selfattention.shape)


#3.构建前向传播神经网络
#两个线性层，中间加了激活函数
class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])#wm,504->2016
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)#wm,2016->504
        self.act_fn = torch.nn.functional.gelu#wm,激活函数
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)#wm,504->2016
        x = self.act_fn(x)#激活函数
        x = self.dropout(x)#wm,丢弃
        x = self.fc2(x)#wm2016->504
        x = self.dropout(x)
        return x

# ##################测试3.MLP模块###########################################################
# mlp=Mlp(config)
# out_mlp=mlp(out_selfattention)
# print("3:out_mlp shape:",out_mlp.shape)


#4.构建编码器的可重复利用的Block()模块：每一个block包含了self-attention模块和MLP模块
class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size#wm,504
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)#wm，层归一化
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

# #############################4.测试Block模块###########################
# block=Block(config,vis=True)
# out_block,_=block(out_embedding)
# print("4:out_block shape:",out_block.shape)


#5.构建Encoder模块，该模块实际上就是堆叠N个Block模块
class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights

# ##################5.测试Encoder模块############################
# encoder=Encoder(config,vis=True)
# out_encoder,_=encoder(out_embedding)
# print("5:out_encoder shape:",out_encoder.shape)


#6构建transformers完整结构，首先图片被embedding模块编码成序列数据，然后送入Encoder中进行编码
class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)#wm,对一幅图片进行切块编码，得到的是（bs,n_patch+1（196）,每一块的维度（768））
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)#wm,输出的是（bs,196,768)
        encoded, attn_weights = self.encoder(embedding_output)#wm,输入的是（bs,196,768)
        return encoded, attn_weights#输出的是（bs,197,768）

# ###################6.测试transformers的完整结构#########################
# #transformers的输入数据是图像数据,输出的数据（bs,197,768）中197的第一个的向量（768）可以用于图像分类
# transformers=Transformer(config,img_size=224,vis=True)
# out_transformers,_=transformers(img)
# print("6:out_transformers shape:",out_transformers.shape)

# 7' data aug
# cutmix
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

# mixup
def mixup_data(image, label, alpha=1.0, use_cuda=False):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = image.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_image = lam * image + (1 - lam) * image[index, :]
    label_a, label_b = label, label[index]
    return mixed_image, label_a, label_b, lam


def mix_criterion(criterion, prediction, label_a, label_b, lam):
    """
    :param criterion: the cross entropy criterion
    :param prediction: y_pred
    :param label_a: label = lam * label_a + (1-lam)* label_b
    :param label_b: label = lam * label_a + (1-lam)* label_b
    :param lam: label = lam * label_a + (1-lam)* label_b
    :return:  cross_entropy(pred,label)
    """
    return lam * criterion(prediction, label_a) + (1 - lam) * criterion(prediction, label_b)


#7构建VisionTransformer，用于图像分类
class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=32, num_classes=100, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier

        self.transformer = Transformer(config, img_size, vis)
        self.head = Linear(config.hidden_size, num_classes)#wm,504-->100

    def forward(self, x, labels=None,method='baseline'):
        if labels is not None:
            # 如果传入真实标签，就直接计算损失值
            if method=='baseline':
                x, attn_weights = self.transformer(x)
                logits = self.head(x[:, 0])
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
                return loss

            elif method=='mixup':
                loss_fct = CrossEntropyLoss()
                mixup_alpha = 1
                mixed_image, label_a, label_b, lam = mixup_data(x, labels, mixup_alpha)
                x, attn_weights = self.transformer(mixed_image)
                logits = self.head(x[:, 0])
                loss = mix_criterion(loss_fct, logits.view(-1, self.num_classes), label_a, label_b, lam)
                return loss

            elif method=='cutmix' or method == 'cutout':
                beta = 1
                cutmix_prob = 0.5
                r = np.random.rand(1)  # 不是每个epoch都进行cutmix
                loss_fct = CrossEntropyLoss()
                if beta > 0 and r < cutmix_prob:
                    # generate mixed sample
                    lam = np.random.beta(beta, beta)
                    rand_index = torch.randperm(x.size()[0])
                    target_a, target_b = labels, labels[rand_index]
                    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
                    if method == 'cutmix':
                        x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
                    else:
                        x[:, :, bbx1:bbx2, bby1:bby2] = 0
                    # adjust lambda to exactly match pixel ratio
                    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
                    # compute output
                    x, attn_weights = self.transformer(x)
                    logits = self.head(x[:, 0])
                    # output = model(inputs)
                    loss = mix_criterion(loss_fct, logits.view(-1, self.num_classes), target_a, target_b, lam)
                else:
                    x, attn_weights = self.transformer(x)
                    logits = self.head(x[:, 0])
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
                return loss
        else:
            x, attn_weights = self.transformer(x)
            logits = self.head(x[:, 0])
            return logits, attn_weights


        # #如果传入真实标签，就直接计算损失值
        # if labels is not None:
        #     loss_fct = CrossEntropyLoss()
        #     loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
        #     return loss
        # else:
        #     return logits, attn_weights



# #################7.验证visionTransformers###########################
# visiontransformer=VisionTransformer(config,num_classes=10,vis=True)
# out_visiontransformer,_=visiontransformer(img)
# print("7.out_visiontransformers shape:",out_visiontransformer.shape)

