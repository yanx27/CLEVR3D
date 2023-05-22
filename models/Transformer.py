#!/usr/bin/env python
# encoding: utf-8
'''
@author: Xu Yan
@file: Transformer.py
@time: 2021/10/26 22:17
'''
import os
import torch
import torch.nn as nn
import warnings
import torchvision.models as models

warnings.filterwarnings('ignore')

from lib.config import CONF
from models.backbone.mlp import MLP
from models.utils import get_siamese_features
from models.base_model import LightningBaseModel
from models.backbone.point_net_pp import PointNetPP
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import RobertaModel, RobertaTokenizerFast
from utils.classes import shape_num, color_num, size_num, material_num
from models.backbone.transformer import MultiLevelEncoder, ScaledDotProductAttention


class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output


class get_model(LightningBaseModel):
    def __init__(self, args, criterion, **kwargs):
        super().__init__(args, criterion)
        self.save_hyperparameters('args')
        self.args = args
        self.relation_type = 'concat'
        self.cnn_name = kwargs.get('cnn_name', 'resnet18')

        d_model = 256
        num_classes = 27
        n_pred_classes = 41
        input_dim = int(args.use_normal) * 3 + int((not args.no_color)) * 3 + int(not args.no_height)
        text_encoder_type = os.path.join(CONF.PATH.DATA, "roberta-base")

        self.tokenizer = RobertaTokenizerFast.from_pretrained(text_encoder_type)
        self.text_encoder = RobertaModel.from_pretrained(text_encoder_type)

        self.resizer = FeatureResizer(
            input_feat_size=768,
            output_feat_size=d_model,
            dropout=args.word_dropout
        )
        self.pre_encoder = PointNetPP(
            sa_n_points=[256, 64, None],
            sa_n_samples=[32, 32, None],
            sa_radii=[0.2, 0.4, None],
            sa_mlps=[[input_dim, 64, 128],
                     [128, 128, 128, 256],
                     [256, 256, 512, d_model]],
            bn=False
        )
        # self.box3d = MLP(6, [64, 128])
        self.object_clf = MLP(d_model, [128, num_classes], 0.3)
        fusion_input = d_model + num_classes + \
                       int(args.use_gt_shape) * shape_num + \
                       int(args.use_gt_color) * color_num + \
                       int(args.use_gt_size) * size_num + \
                       int(args.use_gt_material) * material_num

        self.object_fuse = MLP(fusion_input, [256, d_model])

        if self.args.use_2d:
            if self.cnn_name == 'resnet18':
                self.net2d = models.resnet18(pretrained=True)
            elif self.cnn_name == 'resnet34':
                self.net2d = models.resnet34(pretrained=True)
            self.net2d = nn.Sequential(*list(self.net2d.children())[:-1])
            self.net2d_fuse = nn.Sequential(nn.Dropout(0.5), nn.Linear(512, d_model))

        if self.args.use_scene_graph:
            self.predicate_mlp = MLP(256, [128, 128])
            self.predicate_clf = nn.Conv2d(int(self.relation_type == 'concat') * 128 + 128 + 9, n_pred_classes, 1)

        self.encoder = MultiLevelEncoder(3, d_model=256, h=4, attention_module=ScaledDotProductAttention)

        self.classifer = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, args.answer_classes)
        )

    def _break_up_pc(self, pc):
        # pc may contain color/normals.
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.init_lr)
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)

        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
            'monitor': self.args.monitor_metric
        }

    def forward(self, data_dict):
        device = data_dict['instance_pc'].device
        batch_size, num_instance = data_dict['instance_pc'].shape[:2]

        # extract instance features
        pre_enc_features = get_siamese_features(self.pre_encoder, data_dict['instance_pc'],
                                                aggregator=torch.stack)  # B x N_Objects x object-latent-dim

        # object classification
        objects_cls = get_siamese_features(self.object_clf, pre_enc_features, torch.stack)
        if self.args.use_gt_cls:
            pre_enc_features = torch.cat([pre_enc_features, data_dict['instance_cls_one_hot']], -1)
        else:
            pre_enc_features = torch.cat([pre_enc_features, objects_cls], -1)

        if self.args.use_gt_shape:
            pre_enc_features = torch.cat([pre_enc_features, data_dict['instance_shape'].float()], -1)
        if self.args.use_gt_color:
            pre_enc_features = torch.cat([pre_enc_features, data_dict['instance_color'].float()], -1)
        if self.args.use_gt_size:
            pre_enc_features = torch.cat([pre_enc_features, data_dict['instance_size'].float()], -1)
        if self.args.use_gt_material:
            pre_enc_features = torch.cat([pre_enc_features, data_dict['instance_material'].float()], -1)

        # pre_enc_features = torch.cat([pre_enc_features, box_features], -1)
        pre_enc_features = get_siamese_features(self.object_fuse, pre_enc_features, torch.stack)

        # encode the text
        tokenized = self.tokenizer.batch_encode_plus(
            data_dict['question'],
            max_length=self.args.max_sentence_len,
            truncation=True,
            padding='max_length',
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            encoded_text = self.text_encoder(**tokenized)

        # transpose memory because pytorch's attention expects sequence first
        text_memory = encoded_text.last_hidden_state

        # invert attention mask that we get from huggingface because its the opposite in pytorch transformer
        text_attention_mask = tokenized.attention_mask.ne(1).bool()  # (b, len)

        # resize the encoder hidden states to be of the same d_model as the decoder
        text_memory = self.resizer(text_memory)

        ins_mask = data_dict['instance_mask'].bool()
        features = torch.cat((pre_enc_features, text_memory), dim=1)
        mask = torch.cat((ins_mask, text_attention_mask), dim=1)

        if self.args.use_2d:
            pre_enc_features_2d = get_siamese_features(self.net2d, data_dict['img_sequence'].float(),
                                                       aggregator=torch.stack)  # B x N_Objects x object-latent-dim
            pre_enc_features_2d = get_siamese_features(self.net2d_fuse, pre_enc_features_2d.squeeze(),
                                                       aggregator=torch.stack)  # B x N_Objects x object-latent-dim
            features = torch.cat([features, pre_enc_features_2d], dim=1)
            mask = torch.cat((
                mask,
                torch.ones(
                    pre_enc_features_2d.shape[:2]).to(pre_enc_features_2d.device).bool()
                ), dim=1)

        features = self.encoder(features, attention_mask=mask.unsqueeze(1).unsqueeze(1))

        instance_memory = features[:, :num_instance]  # [B, N_OBJ, D]

        if self.args.use_scene_graph:
            edge_features = data_dict['edge_features']
            predicate_feat = get_siamese_features(self.predicate_mlp, instance_memory, torch.stack)
            predicate_feat_a = predicate_feat.permute(0, 2, 1).unsqueeze(2)
            predicate_feat_b = predicate_feat.permute(0, 2, 1).unsqueeze(3)
            if self.relation_type == 'multiply':
                x_predicate_cls = predicate_feat_b @ predicate_feat_a  # [B, D', N_OBJ, N_OBJ]
            elif self.relation_type == 'concat':
                x_predicate_cls = torch.cat(
                    [predicate_feat_b.repeat(1, 1, 1, num_instance),
                     predicate_feat_a.repeat(1, 1, num_instance, 1)],
                    1
                )  # [B, D'*2, N_OBJ, N_OBJ]
            else:
                raise NotImplementedError

            predicate_cls = self.predicate_clf(
                torch.cat([x_predicate_cls, edge_features.float().permute(0, 3, 1, 2)], 1))
            predicate_cls = predicate_cls.permute(0, 2, 3, 1)
        else:
            predicate_cls = data_dict['edge_labels'].float()

        instance_memory = instance_memory * data_dict['instance_mask'].unsqueeze(-1).repeat(
            (1, 1, instance_memory.shape[-1]))

        text_memory = features[:, num_instance:]
        refer_cls = torch.mean(torch.cat([instance_memory, text_memory], 1), 1)

        if self.args.use_answer_type:
            refer_cls = torch.cat(
                [refer_cls,
                 torch.zeros(batch_size, 13).to(device).scatter_(1, data_dict['answer_type'].view(-1, 1), 1)],
                1)

        refer_cls = self.classifer(refer_cls)

        data_dict['lang_pred'] = refer_cls
        data_dict['objects_cls_pred'] = objects_cls.permute(0, 2, 1)
        data_dict['predicate_predict'] = predicate_cls

        return data_dict


class get_loss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.object_cls_criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.refer_criterion = nn.CrossEntropyLoss()
        self.ssg_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([10]))

    def forward(self, data_dict):
        object_cls_loss = self.object_cls_criterion(
            data_dict['objects_cls_pred'],
            data_dict['instance_cls'],
        )

        if self.args.use_scene_graph:
            predicate_cls_loss = self.ssg_criterion(
                data_dict['predicate_predict'],
                data_dict['edge_labels'].float(),
            )
        else:
            predicate_cls_loss = 0

        refer_loss = self.refer_criterion(data_dict['lang_pred'], data_dict['answer'])

        data_dict['object_cls_loss'] = object_cls_loss
        data_dict['ref_loss'] = refer_loss
        data_dict['ssg_loss'] = predicate_cls_loss
        data_dict['loss'] = refer_loss + object_cls_loss + predicate_cls_loss

        return data_dict
