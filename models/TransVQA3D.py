import os
import time
import torch
import torch.nn as nn
import torchvision

from lib.config import CONF
from models.backbone.mlp import MLP
from models.utils import get_siamese_features
from models.base_model import LightningBaseModel
from models.backbone.point_net_pp import PointNetPP
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.mmt_module import *
from utils.metric import SceneGraphEval
from utils.op_utils import gen_descriptor_batch


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


class TwningAttention(nn.Module):

    def __init__(self, d_model, num_obj):
        super().__init__()
        self.d_model = d_model
        self.num_obj = num_obj
        self.row_conv = nn.Sequential(nn.Conv2d(d_model, d_model, 1), nn.LayerNorm([d_model, num_obj, num_obj]),
                                      nn.ReLU(inplace=True), nn.Conv2d(d_model, d_model, [num_obj, 1]))
        self.column_conv = nn.Sequential(nn.Conv2d(d_model, d_model, 1), nn.LayerNorm([d_model, num_obj, num_obj]),
                                         nn.ReLU(inplace=True), nn.Conv2d(d_model, d_model, [1, num_obj]))
        self.node_conv = nn.Sequential(nn.Conv1d(d_model, d_model, 1), nn.LayerNorm([d_model, num_obj]), nn.ReLU(inplace=True))

        self.edge_conv = nn.Sequential(nn.Conv2d(d_model * 2, d_model, 1), nn.LayerNorm([d_model, num_obj, num_obj]),
                                       nn.ReLU(inplace=True))

    def forward(self, edge_feat, node_feat, edge_mask):
        '''
        @param edge_feat:  B, D, M, M
        @param node_feat:  B, L1, D
        @param edge_mask:  B, M, M
        '''
        edge_feat = edge_mask.unsqueeze(1) * edge_feat
        row_feat = self.row_conv(edge_feat).squeeze(2)
        column_feat = self.column_conv(edge_feat).squeeze(3)
        weights = nn.functional.sigmoid(row_feat * column_feat)
        new_node_feat = self.node_conv(node_feat.permute(0, 2, 1) * weights)

        edge_a = new_node_feat.unsqueeze(2)
        edge_b = new_node_feat.unsqueeze(3)

        new_edge_feat = self.edge_conv(
            torch.cat([edge_b.repeat(1, 1, 1, self.num_obj),
                       edge_a.repeat(1, 1, self.num_obj, 1)], 1))     # [B, D, N_OBJ, N_OBJ]

        return new_node_feat.permute(0, 2, 1), new_edge_feat


class SGAA(nn.Module):

    def __init__(self, d_model, num_obj, num_twn=2):
        super().__init__()
        self.d_model = d_model
        self.num_obj = num_obj
        self.num_twn = num_twn
        self.twn_att = nn.ModuleList()
        self.node_mlps = nn.ModuleList()
        self.edge_mlps = nn.ModuleList()

        self.language_mlp = nn.Sequential(nn.Conv1d(d_model, d_model, 1), nn.ReLU(inplace=True))
        for i in range(num_twn):
            self.twn_att.append(TwningAttention(d_model, num_obj))
            self.node_mlps.append(nn.Sequential(nn.Conv1d(d_model, d_model, 1), nn.LayerNorm([d_model, num_obj])))
            self.edge_mlps.append(nn.Sequential(nn.Conv2d(d_model, d_model, 1), nn.LayerNorm([d_model, num_obj, num_obj])))

        self.x_attention = nn.MultiheadAttention(d_model, num_heads=4)

    def forward(self, edge_feat, node_feat, lang_feat, edge_mask):
        '''
        @param edge_feat:  B, D, M, M
        @param edge_mask:  B, M, M
        @param node_feat:  B, L1, D
        @param lang_feat:  B, L2, D
        '''

        # Twning Attention
        for i in range(self.num_twn):
            node_feat, edge_feat = self.twn_att[i](edge_feat, node_feat, edge_mask)
            node_feat = self.node_mlps[i](node_feat.permute(0, 2, 1)).permute(0, 2, 1)
            edge_feat = self.edge_mlps[i](edge_feat)

        lang_feat = self.language_mlp(lang_feat.permute(0, 2, 1)).permute(2, 0, 1)
        node_feat = node_feat.permute(1, 0, 2)

        lang_feat, _ = self.x_attention(lang_feat, node_feat, node_feat)
        lang_feat = lang_feat.permute(1, 0, 2)

        return edge_feat, node_feat, lang_feat


class get_model(LightningBaseModel):

    def __init__(self, args, criterion, **kwargs):
        super().__init__(args, criterion)
        self.save_hyperparameters('args')
        self.args = args
        self.relation_type = 'concat'
        self.cnn_name = kwargs.get('cnn_name', 'resnet18')
        self.no_vision = args.no_vision

        d_model = 768
        num_classes = kwargs.get('num_obj_class', 27)
        n_pred_classes = 27
        input_dim = int(args.use_normal) * 3 + int((not args.no_color)) * 3 + int(not args.no_height)

        self.pre_encoder = PointNetPP(sa_n_points=[256, 64, None],
                                      sa_n_samples=[32, 32, None],
                                      sa_radii=[0.2, 0.4, None],
                                      sa_mlps=[[input_dim, 64, 128], [128, 128, 128, 256], [256, 256, 512, d_model]],
                                      bn=False)
        # self.box3d = MLP(6, [64, 128])
        self.object_clf = MLP(d_model, [128, num_classes], 0.3)
        self.object_fuse = MLP(d_model, [256, d_model])

        if self.args.use_2d:
            if self.cnn_name == 'resnet18':
                self.net2d = torchvision.models.resnet18(pretrained=True)
            elif self.cnn_name == 'resnet34':
                self.net2d = torchvision.models.resnet34(pretrained=True)
            self.net2d = nn.Sequential(*list(self.net2d.children())[:-1])
            self.net2d_fuse = nn.Sequential(nn.Dropout(0.5), nn.Linear(512, d_model))

        if self.args.use_scene_graph:
            self.sga_att = SGAA(d_model, args.max_instance_num, num_twn=2)
            self.predicate_mlp = MLP(d_model, [128, 128])
            self.predicate_fuse = nn.Conv2d(int(self.relation_type == 'concat') * 128 + 128 + 9, d_model, 1)
            self.predicate_clf = nn.Conv2d(d_model, n_pred_classes, 1)
            self.object_clf_refine = MLP(d_model, [128, num_classes], 0.3)

        text_encoder_type = os.path.join(CONF.PATH.HF, 'bert-base-uncased')
        # Encoders for text
        self.text_bert_config = BertConfig(hidden_size=d_model, num_hidden_layers=3, num_attention_heads=12, type_vocab_size=2)
        self.text_bert = TextBert.from_pretrained(text_encoder_type, config=self.text_bert_config)

        self.mmt_config = BertConfig(hidden_size=d_model, num_hidden_layers=4, num_attention_heads=12, type_vocab_size=2)
        self.mmt = MMT(self.mmt_config)

        self.linear_obj_feat_to_mmt_in = nn.Linear(d_model, d_model)
        self.linear_obj_bbox_to_mmt_in = nn.Linear(11, d_model)
        self.obj_feat_layer_norm = nn.LayerNorm(d_model)
        self.obj_bbox_layer_norm = nn.LayerNorm(d_model)
        self.obj_drop = nn.Dropout(0.1)

        self.classifer = nn.Sequential(nn.ReLU(),
                                       nn.Linear(d_model + int(args.use_answer_type) * len(self.val_class_acc.ref_dict), 256),
                                       nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.4), nn.Linear(256, args.answer_classes))

    def _break_up_pc(self, pc):
        # pc may contain color/normals.
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def configure_optimizers(self):
        backbone_name = []
        backbone_name.append('text_bert.')     # exclude text_bert_out_linear
        backbone_param, rest_param = [], []
        for kv in self.named_parameters():
            isbackbone = [int(key in kv[0]) for key in backbone_name]
            if sum(isbackbone + [0]):
                backbone_param.append(kv[1])
            else:
                rest_param.append(kv[1])
        optimizer = torch.optim.Adam([{
            'params': rest_param
        }, {
            'params': backbone_param,
            'lr': self.args.init_lr / 10.
        }],
                                     lr=self.args.init_lr)
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler, 'monitor': self.args.monitor_metric}

    def forward(self, data_dict):
        # extract instance features
        pre_enc_features = get_siamese_features(self.pre_encoder, data_dict['instance_pc'],
                                                aggregator=torch.stack)     # B x N_Objects x object-latent-dim
        # object classification
        objects_cls = get_siamese_features(self.object_clf, pre_enc_features, torch.stack)
        if self.args.use_gt_cls:
            pre_enc_features = torch.cat([pre_enc_features, data_dict['instance_cls_one_hot']], -1)

        # Get feature for utterance
        txt_inds = data_dict["token_inds"]     # batch_size, lang_size
        txt_type_mask = torch.ones(txt_inds.shape, device=torch.device('cuda')) * 1.
        txt_mask = _get_mask(data_dict['token_num'].to(txt_inds.device), txt_inds.size(1))     ## all proposals are non-empty
        txt_type_mask = txt_type_mask.long()

        txt_emb = self.text_bert(txt_inds=txt_inds, txt_mask=txt_mask, txt_type_mask=txt_type_mask)

        obj_mmt_in = self.obj_feat_layer_norm(self.linear_obj_feat_to_mmt_in(pre_enc_features)) + \
            self.obj_bbox_layer_norm(self.linear_obj_bbox_to_mmt_in(gen_descriptor_batch(data_dict['instance_pc'][..., :3])))

        obj_mmt_in = self.obj_drop(obj_mmt_in)
        obj_num = obj_mmt_in.size(1)
        obj_mask = _get_mask(data_dict['context_size'], obj_num)     ## all proposals are non-empty

        if self.args.use_2d:
            pre_enc_features_2d = get_siamese_features(self.net2d, data_dict['img_sequence'].float(),
                                                       aggregator=torch.stack)     # B x N_Objects x object-latent-dim
            pre_enc_features_2d = get_siamese_features(self.net2d_fuse, pre_enc_features_2d.squeeze(),
                                                       aggregator=torch.stack)     # B x N_Objects x object-latent-dim
            obj_mmt_in = torch.cat([obj_mmt_in, pre_enc_features_2d], dim=1)
            obj_mask = torch.cat((obj_mask, torch.ones(pre_enc_features_2d.shape[:2]).to(pre_enc_features_2d.device).bool()),
                                 dim=1)

        mmt_results = self.mmt(txt_emb=txt_emb, txt_mask=txt_mask, obj_emb=obj_mmt_in, obj_mask=obj_mask, obj_num=obj_num)

        instance_memory = mmt_results['mmt_obj_output']     # [B, N_OBJ, D]
        text_memory = mmt_results['mmt_txt_output']

        if self.args.use_scene_graph:
            edge_features = data_dict['edge_features']
            predicate_feat = get_siamese_features(self.predicate_mlp, instance_memory, torch.stack)
            predicate_feat_a = predicate_feat.permute(0, 2, 1).unsqueeze(2)
            predicate_feat_b = predicate_feat.permute(0, 2, 1).unsqueeze(3)
            if self.relation_type == 'multiply':
                x_predicate_cls = predicate_feat_b @ predicate_feat_a     # [B, D', N_OBJ, N_OBJ]
            elif self.relation_type == 'concat':
                x_predicate_cls = torch.cat(
                    [predicate_feat_b.repeat(1, 1, 1, obj_num),
                     predicate_feat_a.repeat(1, 1, obj_num, 1)], 1)     # [B, D'*2, N_OBJ, N_OBJ]
            else:
                raise NotImplementedError

            edge_feat = torch.cat([x_predicate_cls, edge_features.float().permute(0, 3, 1, 2)], 1)

            edge_feat = self.predicate_fuse(edge_feat)
            edge_mask = data_dict['edge_mask'].float()
            edge_feat, instance_memory, _text_memory = self.sga_att(edge_feat, instance_memory, text_memory, edge_mask)
            instance_memory = instance_memory.permute(1, 0, 2)
            predicate_cls = self.predicate_clf(edge_feat).permute(0, 2, 3, 1)
            # objects_cls = objects_cls + get_siamese_features(self.object_clf_refine, instance_memory, torch.stack)
            text_memory = text_memory + _text_memory
        else:
            predicate_cls = data_dict['edge_labels'].float()
        instance_memory = instance_memory * data_dict['instance_mask'].unsqueeze(-1).repeat((1, 1, instance_memory.shape[-1]))

        refer_cls = text_memory.max(1)[0]

        refer_cls = self.classifer(refer_cls)

        data_dict['lang_pred'] = refer_cls
        data_dict['objects_cls_pred'] = objects_cls.permute(0, 2, 1)
        data_dict['predicate_predict'] = predicate_cls

        return data_dict


class get_loss(nn.Module):

    def __init__(self, args, ref_loss='ce'):
        super().__init__()
        self.args = args
        self.ref_loss = ref_loss
        self.object_cls_criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.object_usable_criterion = nn.CrossEntropyLoss(ignore_index=-1)
        if self.ref_loss == 'ce':
            self.refer_criterion = nn.CrossEntropyLoss()
        elif self.ref_loss == 'bce':
            self.refer_criterion = torch.nn.BCELoss(reduction='sum')
        else:
            raise NotImplementedError
        self.ssg_criterion = nn.BCEWithLogitsLoss(reduction='none')

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
            predicate_cls_loss *= data_dict['edge_mask'].unsqueeze(-1)
            predicate_cls_loss = torch.mean(predicate_cls_loss)
        else:
            predicate_cls_loss = 0

        if self.ref_loss == 'ce':
            refer_loss = self.refer_criterion(data_dict['lang_pred'], data_dict['answer'])
        elif self.ref_loss == 'bce':
            ref_label = torch.zeros(data_dict['lang_pred'].shape[0],
                                    self.args.answer_classes).to(data_dict['lang_pred'].device).scatter_(
                                        1, data_dict['answer'].view(-1, 1), 1)
            refer_loss = self.refer_criterion(data_dict['lang_pred'], ref_label)
        else:
            raise NotImplementedError

        data_dict['object_cls_loss'] = object_cls_loss
        data_dict['ref_loss'] = refer_loss
        data_dict['ssg_loss'] = predicate_cls_loss
        data_dict['loss'] = refer_loss + object_cls_loss + predicate_cls_loss

        return data_dict


## pad at the end; used anyway by obj, ocr mmt encode
def _get_mask(nums, max_num):
    # non_pad_mask: b x lq, torch.float32, 0. on PAD
    batch_size = nums.size(0)
    arange = torch.arange(0, max_num).unsqueeze(0).expand(batch_size, -1)
    non_pad_mask = arange.to(nums.device).lt(nums.unsqueeze(-1))
    non_pad_mask = non_pad_mask.type(torch.float32)
    return non_pad_mask
