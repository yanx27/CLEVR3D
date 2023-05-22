#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: Xu Yan
@File: base_model.py
@Time: 2021/10/24 18:07
'''

import torch
import pytorch_lightning as pl

from pytorch_lightning.metrics import Accuracy
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import ClfAccuracy, ClassAccuracy, SceneGraphEval

class LightningBaseModel(pl.LightningModule):
    def __init__(self, args, criterion):
        super().__init__()
        self.args = args
        self.criterion = criterion
        self.train_ref_acc = Accuracy()
        self.train_ssg_acc = Accuracy()
        self.train_cls_acc = ClfAccuracy(-1)

        self.val_ref_acc = Accuracy(compute_on_step=False)
        self.val_class_acc = ClassAccuracy(compute_on_step=False)
        self.val_cls_acc = ClfAccuracy(-1, compute_on_step=False)
        self.val_ssg_metric = SceneGraphEval(compute_on_step=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.init_lr)
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10, verbose=True)

        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
            'monitor': self.args.monitor_metric
        }

    def forward(self, data):
        pass

    def training_step(self, data_dict, batch_idx):
        data_dict = self.forward(data_dict)
        data_dict = self.criterion(data_dict)
        lang_choice = torch.argmax(data_dict['lang_pred'], 1)
        cls_choice = torch.argmax(data_dict['objects_cls_pred'], 1)

        self.train_cls_acc(cls_choice, data_dict['instance_cls'])
        self.train_ref_acc(lang_choice, data_dict['answer'])

        self.log('train/object_cls_loss', data_dict['object_cls_loss'], on_epoch=True)
        self.log('train/ref_loss', data_dict['ref_loss'], on_epoch=True)
        self.log('train/ref_acc', self.train_ref_acc, on_epoch=True)
        self.log('train/cls_acc', self.train_cls_acc, on_epoch=True)

        if self.args.use_scene_graph:
            self.log('train/ssg_loss', data_dict['ssg_loss'])

        total_loss = data_dict['loss']

        return total_loss

    def validation_step(self, data_dict, batch_idx):
        data_dict = self.forward(data_dict)
        data_dict = self.criterion(data_dict)

        answer_type = data_dict['answer_type_str']
        lang_choice = torch.argmax(data_dict['lang_pred'], 1)
        cls_choice = torch.argmax(data_dict['objects_cls_pred'], 1)

        self.val_cls_acc(cls_choice, data_dict['instance_cls'])
        self.val_ref_acc(lang_choice, data_dict['answer'])
        self.val_class_acc(lang_choice, data_dict['answer'], answer_type)

        if self.args.use_scene_graph:
            self.val_ssg_metric(data_dict)

        # self.log('val/usable_loss', data_dict['usable_loss'], on_epoch=True)
        self.log('val/object_cls_loss', data_dict['object_cls_loss'], on_epoch=True)
        self.log('val/ref_loss', data_dict['ref_loss'], on_epoch=True)
        self.log('val/ref_acc', self.val_ref_acc, on_epoch=True)
        self.log('val/cls_acc', self.val_cls_acc, on_epoch=True)

        if self.args.use_scene_graph:
            self.log('val/ssg_loss', data_dict['ssg_loss'])


    def validation_epoch_end(self, outputs):
        str_print = ''
        if self.args.use_scene_graph:
            ssg_output = self.val_ssg_metric.compute()
            for k, v in list(ssg_output.items()):
                str_print += '\n[{}]: {}'.format(k, v)
        str_print += '\n-----------------'

        class_output = self.val_class_acc.compute()
        for k in list(class_output.keys()):
            str_print +='\n[{}]: {:.3f} ({}/{})'.format(
                        k,
                        class_output[k][0],
                        class_output[k][1],
                        class_output[k][2],
                    )
        str_print += '\n-----------------'

        str_print += '\nThe class accuracy: {:.3f}'.format(self.val_cls_acc.compute())
        str_print += '\nThe total refer accuracy: {:.3f}'.format(self.val_ref_acc.compute())
        self.print(str_print)
