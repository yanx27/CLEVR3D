import torch
import pytorch_lightning as pl

from torchmetrics.classification.accuracy import Accuracy
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import ClfAccuracy, ClassAccuracy, SceneGraphEval
from transformers import get_linear_schedule_with_warmup
import time


class LightningBaseModel(pl.LightningModule):

    def __init__(self, args, criterion):
        super().__init__()
        self.args = args
        self.criterion = criterion
        self.train_ref_acc = Accuracy(task='multiclass', num_classes=args.answer_classes)
        self.train_cls_acc = ClfAccuracy(-1)

        self.val_ref_acc = Accuracy(task='multiclass', num_classes=args.answer_classes)
        self.val_class_acc = ClassAccuracy()
        self.val_cls_acc = ClfAccuracy(-1)

    def configure_optimizers(self):
        self.trainer.model.parameters()
        self.trainer.reset_train_dataloader()
        train_dataloader = self.trainer.train_dataloader
        batch_per_epoch = len(train_dataloader)
        t_total = batch_per_epoch * self.trainer.max_epochs
        # warmup_ratio = self.optim_cfg.warmup_ratio
        warmup_ratio = 0.1
        warmup_iters = int(t_total * warmup_ratio)

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.args.init_lr)

        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_iters, t_total)
        return {'optimizer': optimizer, 'lr_scheduler': {"scheduler": scheduler, "interval": "step"}}

    def forward(self, data):
        pass

    def training_step(self, data_dict, batch_idx):
        data_dict = self.forward(data_dict)
        data_dict = self.criterion(data_dict)
        lang_choice = torch.argmax(data_dict['lang_pred'], 1)
        cls_choice = torch.argmax(data_dict['objects_cls_pred'], 1)

        self.train_cls_acc(cls_choice, data_dict['instance_cls'])
        self.train_ref_acc(lang_choice, data_dict['answer'])

        # self.log('train/usable_loss', data_dict['usable_loss'], on_epoch=True)
        # self.log('train/object_cls_loss', data_dict['object_cls_loss'], on_epoch=True)
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

        self.val_cls_acc.update(cls_choice, data_dict['instance_cls'])
        self.val_ref_acc.update(lang_choice, data_dict['answer'])
        self.val_class_acc.update(lang_choice, data_dict['answer'], answer_type)

        # if self.args.use_scene_graph:
        #     self.val_ssg_metric(data_dict)

        self.log('val/ref_loss', data_dict['ref_loss'], sync_dist=True)
        self.log('val/ref_acc', self.val_ref_acc, sync_dist=True)
        self.log('val/cls_acc', self.val_cls_acc, sync_dist=True)

        if self.args.use_scene_graph:
            self.log('val/ssg_loss', data_dict['ssg_loss'])

        return {
            'question': data_dict['question'],
            'label': data_dict['answer'],
            'pred': lang_choice,
            'answer_type': data_dict['answer_type'],
            'template': data_dict['template'],
        }

    def validation_epoch_end(self, outputs):
        str_print = ''

        str_print += '\n-----------------'

        class_output = self.val_class_acc.compute()
        for k in list(class_output.keys()):
            str_print += '\n[{}]: {:.3f} ({}/{})'.format(
                k,
                class_output[k][0],
                class_output[k][1],
                class_output[k][2],
            )
        str_print += '\n-----------------'

        str_print += '\nThe class accuracy: {:.3f}'.format(self.val_cls_acc.compute())
        str_print += '\nThe total refer accuracy: {:.3f}'.format(self.val_ref_acc.compute())
        self.print(str_print)

        rel_correct = 0
        nonrel_correct = 0
        rel_total = 0
        nonrel_total = 0
        for output in outputs:
            for i in range(len(output['question'])):
                if output['template'][i] in ['zero_hop', 'single_or']:
                    if output['pred'][i] == output['label'][i]:
                        rel_correct += 1
                    rel_total += 1
                else:
                    if output['pred'][i] == output['label'][i]:
                        nonrel_correct += 1
                    nonrel_total += 1

        self.log('val/rel_acc', rel_correct / rel_total, sync_dist=True)
        self.log('val/nonrel_acc', nonrel_correct / nonrel_total, sync_dist=True)
