#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: Xu Yan
@File: R3ScanDataset.py
@Time: 2021/10/23 9:36
'''
import os
import cv2
import math
import time
import json
import torch
import pickle
import imageio
import numpy as np

from lib.config import CONF
from torch.utils.data import Dataset
from data.R3Scan.r3scan_utils import random_point_dropout, shuffle_points
from utils.classes import shape, color, size, material

GLOVE_PICKLE = os.path.join(CONF.PATH.DATA, "glove.p")


class R3ScanVQAO27R16Dataset(Dataset):
    def __init__(self,
                 clever3d,
                 r3scan_scene,
                 split="train",
                 num_points=4096,
                 use_height=False,
                 use_color=False,
                 use_normal=False,
                 use_scene_graph=False,
                 augment=False,
                 no_vision=False,
                 **kwargs):

        self.clever3d = clever3d
        self.r3scan_all_scene = r3scan_scene
        self.split = split
        self.no_vision = no_vision
        self.use_color = use_color
        self.use_height = use_height
        self.use_normal = use_normal
        self.use_scene_graph = use_scene_graph
        self.augment = augment
        self.preloading = kwargs.get('preloading', False)

        self.image_path = '/dataset/3rscan/{}/sequence'
        self.max_img_num = 20
        self.image_dim = (256, 256)
        self.instance_pt_num = num_points
        self.max_instance_num = kwargs.get('max_instance_num', 64)
        self.max_sentence_len = kwargs.get('max_sentence_len', 64)
        self.first_run = kwargs.get('first_run', False)
        self.use_2d = kwargs.get('use_2d', False)
        self.num_ralationships = kwargs.get('num_ralationships', 16)

        scene_list_3dssg = []
        scene_list_3dssg += np.loadtxt(os.path.join(CONF.PATH.R3SCAN_META, 'train_scans.txt'), dtype='str').tolist()
        scene_list_3dssg += np.loadtxt(os.path.join(CONF.PATH.R3SCAN_META, 'validation_scans.txt'),
                                       dtype='str').tolist()
        scene_list_3dssg += np.loadtxt(os.path.join(CONF.PATH.R3SCAN_META, 'test_scans.txt'), dtype='str').tolist()
        self.scene_list_3dssg = scene_list_3dssg

        # load data
        self._load_data()

        self.answer_type_dict = {'count': 0,
                                 'equal_color': 1,
                                 'equal_integer': 2,
                                 'equal_material': 3,
                                 'equal_shape': 4,
                                 'equal_size': 5,
                                 'exist': 6,
                                 'greater_than': 7,
                                 'less_than': 8,
                                 'query_color': 9,
                                 'query_material': 10,
                                 'query_shape': 11,
                                 'query_size': 12,
                                 'query_label': 13
                                 }
        self.question_type_dict = {'count': 0,
                                   'equal_color': 1,
                                   'equal_integer': 1,
                                   'equal_material': 1,
                                   'equal_shape': 1,
                                   'equal_size': 1,
                                   'exist': 1,
                                   'greater_than': 1,
                                   'less_than': 1,
                                   'query_color': 2,
                                   'query_material': 2,
                                   'query_shape': 2,
                                   'query_size': 2,
                                   'query_label': 2
                                   }

    def __len__(self):
        return len(self.question)

    def _tranform_des(self):
        with open(GLOVE_PICKLE, "rb") as f:
            glove = pickle.load(f)

        valid_scene_name = []
        all_lang_questions = []
        all_lang_questions_embedding = []
        all_lang_answers = []
        all_lang_answers_type = []

        for data in self.clever3d:
            scene_id = data["scan"]

            if scene_id in self.r3scan_all_scene and scene_id in self.scene_list_3dssg:
                valid_scene_name.append(scene_id)
                question = data["question"]
                answer = data["answer"]
                tokens = question.strip('?').split(' ')

                # tokenize the description
                embeddings = np.zeros((self.max_sentence_len, 300))
                for token_id in range(self.max_sentence_len):
                    if token_id < len(tokens):
                        token = tokens[token_id]
                        if token in glove:
                            embeddings[token_id] = glove[token]
                        else:
                            embeddings[token_id] = glove["unk"]

                # store
                all_lang_questions.append({'scans': scene_id, 'question': question})
                all_lang_questions_embedding.append(embeddings)
                all_lang_answers.append(answer)
                all_lang_answers_type.append(data["question_type"])

        data_file_name = str(CONF.PATH.CLEVER3D_data).split('/')[-1].replace('.json', '')
        data_dict_name = os.path.join(CONF.PATH.CLEVER3D, 'answer_dict_%s.json' % data_file_name)
        if os.path.exists(data_dict_name):
            print('load answer_dict.json...')
            with open(data_dict_name, ) as f:
                answer_dict = json.load(f)
            if self.split == 'train':
                print(answer_dict)
        else:
            lang_answers_list = list(set([str(d['answer']) for d in self.clever3d]))
            lang_answers_sort = {}
            exist_list, query_list = [], []

            number_list = list(range(30))
            for ans in lang_answers_list:
                if ans in ['True', 'False']:
                    exist_list.append(ans)
                else:
                    if ans not in [str(i) for i in number_list]:
                        query_list.append(ans)

            sorted_answers = sorted(number_list) + list(set(exist_list)) + list(set(query_list))

            for v, k in enumerate(sorted_answers):
                lang_answers_sort[str(k)] = v

            if self.split == 'train':
                with open(data_dict_name, 'w') as f:
                    json.dump(lang_answers_sort, f)
                print(lang_answers_sort)

            answer_dict = lang_answers_sort

        self.answer_dict = answer_dict
        self.answer_classes_num = len(answer_dict)
        self.valid_scene_name = valid_scene_name
        print('Total %d %s questions in %d scenes...' % (
            len(self.valid_scene_name), self.split, len(set(self.valid_scene_name))))

        return all_lang_questions, all_lang_questions_embedding, all_lang_answers, all_lang_answers_type

    def _load_data(self):
        print("loading %s language data..." % self.split)
        # load language features
        self.question, self.question_emb, self.answer, self.answer_type = self._tranform_des()
        self.num_answer_class = len(self.answer)

        print('loading %s scene graph...' % self.split)
        self.scene_graph = json.load(open(os.path.join(CONF.PATH.R3SCAN, "SceneGraphAnnotation.json")))

        self.scene_list = sorted(list(set([d['scans'] for d in self.question if d['scans'] in self.r3scan_all_scene])))
        self.scene_data = {}
        if not self.no_vision and self.preloading:
            print("loading %s point clouds..." % self.split)
            for scene_id in self.scene_list:
                self.scene_data[scene_id] = {}
                self.scene_data[scene_id]["mesh_vertices"] = np.load(
                    os.path.join(CONF.PATH.R3SCAN, 'R3Scan_noSample_27cls', scene_id) + "_vert.npy")
                self.scene_data[scene_id]["instance_labels"] = np.load(
                    os.path.join(CONF.PATH.R3SCAN, 'R3Scan_noSample_27cls', scene_id) + "_ins_label.npy")

    def __getitem__(self, idx):
        start = time.time()

        # get language features
        question = self.question[idx]
        lang_feat = self.question_emb[idx]
        lang_len = len(question['question'].strip('?').split(' '))
        lang_len = lang_len if lang_len <= self.max_sentence_len else self.max_sentence_len
        answer_label = self.answer_dict[str(self.answer[idx])]
        answer_label_type = self.answer_type_dict[self.answer_type[idx]]
        question_label_type = self.question_type_dict[self.answer_type[idx]]

        img_sequence = np.zeros((self.max_img_num, self.image_dim[0], self.image_dim[1], 3))
        if not self.no_vision:
            # get images
            if self.use_2d:
                image_list = [x for x in sorted(os.listdir(self.image_path.format(question['scans']))) if
                              x.find('.jpg') != -1]
                image_list = [x for (i, x) in enumerate(image_list) if i % 20 == 0]
                imgs = []
                for img_path in image_list:
                    img = imageio.imread(os.path.join(self.image_path.format(question['scans']), img_path))
                    img = cv2.resize(img, (self.image_dim[0], self.image_dim[1]), interpolation=cv2.INTER_NEAREST)
                    imgs.append(img)
                imgs = np.stack(imgs, 0)
                img_num = len(imgs)
                if img_num > self.max_img_num:
                    img_sequence = imgs[: self.max_img_num]
                else:
                    img_sequence[:img_num] = imgs

            # get pc
            if self.preloading:
                mesh_vertices = self.scene_data[question['scans']]["mesh_vertices"].copy()
                instance_labels = self.scene_data[question['scans']]["instance_labels"].copy()
            else:
                mesh_vertices = np.load(
                    os.path.join(CONF.PATH.R3SCAN, 'R3Scan_noSample_27cls', question['scans']) + "_vert.npy")
                instance_labels = np.load(
                    os.path.join(CONF.PATH.R3SCAN, 'R3Scan_noSample_27cls', question['scans']) + "_ins_label.npy")
            if not self.use_color:
                point_cloud = mesh_vertices[:, 0:3]  # do not use color for now
            else:
                point_cloud = mesh_vertices[:, 0:6]
                point_cloud[:, 3:6] = point_cloud[:, 3:6] / 256

            if self.use_normal:
                normals = mesh_vertices[:, 6:9]
                point_cloud = np.concatenate([point_cloud, normals], 1)

            if self.use_height:
                floor_height = np.percentile(point_cloud[:, 2], 0.99)
                height = point_cloud[:, 2] - floor_height
                point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)], 1)

            if self.augment and self.split == 'train':
                # randomly rotate points
                theta = np.random.rand() * 2 * math.pi
                point_cloud[:, :3] = np.matmul(point_cloud[:, :3],
                                               [[math.cos(theta), math.sin(theta), 0],
                                                [-math.sin(theta), math.cos(theta), 0],
                                                [0, 0, 1]])
                # randomly dropout
                point_cloud[:, :3] = random_point_dropout(point_cloud[:, :3])

                # randomly shuffle
                point_cloud[:, :3] = shuffle_points(point_cloud[:, :3])

            # from data.R3Scan.visualize import write_obj
            # os.makedirs('visual', exist_ok=True)
            # write_obj(point_cloud, 'visual/pc', rgb=True)

            # get scene graph
            obj_id_list = []
            labels = {}
            attribute = {}

            for instance_id, object in self.scene_graph[question['scans']]['nodes'].items():
                obj_id_list.append(int(instance_id))
                labels[int(instance_id)] = int(object['rio27_enc'])

            # dict contains {object_id: point cloud}
            objects_dict = {int(idx): point_cloud[instance_labels[:, 0] == idx, :] \
                            for idx in list(np.unique(instance_labels)) if idx in obj_id_list}
        else:
            point_cloud = np.zeros((4000, 3))

        point_cloud = point_cloud[np.random.choice(point_cloud.shape[0],
                                                   40000,
                                                   replace=point_cloud.shape[0] < 40000)]

        # prepare point cloud instances
        num_instance = self.max_instance_num
        instance_points = np.zeros((num_instance, self.instance_pt_num, point_cloud.shape[-1]), dtype=np.float32)
        instance_centers = np.zeros((num_instance, 3), dtype=np.float32)
        instance_indices = np.ones(num_instance, dtype=np.int64) * -1
        instance_descriptor = np.ones((num_instance, 11), dtype=np.float32)
        instance_map = {}
        instance_mask = np.zeros(num_instance, dtype=np.int64)
        instance_semantics = np.ones(num_instance, dtype=np.int64) * -1
        instance_semantics_one_hot = np.zeros((num_instance, 27), dtype=np.int64)
        num_valid_instance = 0

        if not self.no_vision:
            for k, pc in objects_dict.items():
                if labels[k] not in [0, 1, 2, 15, 26]:  # exclude  '-' (0),  wall (1), floor (2), ceiling (15) objects (26) [0, 1, 2, 15, 26]
                    instance_indices[num_valid_instance] = k
                    instance_mask[num_valid_instance] = 1
                    instance_map[num_valid_instance] = k
                    instance_semantics[num_valid_instance] = labels[k] - 1
                    instance_semantics_one_hot[num_valid_instance, labels[k] - 1] = 1

                    center = 0.5 * (pc.min(0) + pc.max(0))
                    pc_size = pc.max(0) - pc.min(0)
                    instance_centers[num_valid_instance] = center[:3]

                    # randomly sample points
                    if pc.shape[0] >= self.instance_pt_num:
                        choices = np.random.choice(pc.shape[0], self.instance_pt_num, replace=False)
                    else:
                        choices = np.random.choice(pc.shape[0], self.instance_pt_num, replace=True)
                    instance_points[num_valid_instance] = pc[choices]
                    instance_descriptor[num_valid_instance, :3] = center[:3]
                    instance_descriptor[num_valid_instance, 3:6] = pc_size[:3]
                    instance_descriptor[num_valid_instance, 6:9] = pc[:, :3].std(0)
                    instance_descriptor[num_valid_instance, 9] = pc_size[0] * pc_size[1] * pc_size[2]
                    instance_descriptor[num_valid_instance, 10] = pc_size.max()

                    num_valid_instance += 1
                    if num_valid_instance >= self.max_instance_num:
                        break

        instance_descriptor[instance_descriptor==0] = 1
        objects_id = instance_indices[instance_indices != -1]
        edge_labels = np.zeros((self.max_instance_num, self.max_instance_num, self.num_ralationships))
        edge_mask = np.zeros((self.max_instance_num, self.max_instance_num))
        edge_features = np.zeros((self.max_instance_num, self.max_instance_num, 9))
        triples = []
        pairs = []
        edges = []

        if self.use_scene_graph:
            # prepare edge feature and labels
            for triple in self.scene_graph[question['scans']]['edges']:
                if (int(triple[0]) not in objects_id) or (int(triple[1]) not in objects_id) or (triple[0] == triple[1]):
                    continue

                # indices of triple[0], triple[1] in `objects_id`
                obj1 = np.where(objects_id == int(triple[0]))[0][0]
                obj2 = np.where(objects_id == int(triple[1]))[0][0]
                triples.append([obj1, obj2, int(triple[2])])
                tmp_pair = [obj1, obj2]
                if tmp_pair not in pairs:
                    pairs.append(tmp_pair)

            for triple in triples:
                edge_mask[triple[0], triple[1]] = 1
                edge_labels[triple[0], triple[1], triple[2]] = 1

            pairs = np.array(pairs).reshape((-1, 2))
            triples = np.array(triples).reshape((-1, 3))
            valid_index = np.all(pairs < self.max_instance_num, 1)
            pairs = pairs[valid_index]
            triples = triples[valid_index]

            for i in range(self.max_instance_num):
                for j in range(self.max_instance_num):
                    center_offset = instance_centers[i] - instance_centers[j]
                    center_obj = instance_centers[i]
                    center_sub = instance_centers[j]
                    edge_features[i, j] = np.concatenate([center_offset, center_obj, center_sub], 0)

            s, o = np.split(np.array(pairs), 2, axis=1)  # All have shape (T, 1)
            s, o = [np.squeeze(x, axis=1) for x in [s, o]]  # Now have shape (T,)
            for index, v in enumerate(s):
                s[index] = v  # s_idx
            for index, v in enumerate(o):
                o[index] = v  # o_idx
            edges = np.stack((s, o), axis=1)

        data_dict = {}

        # language
        data_dict['scene'] = question['scans']
        data_dict['tokens'] = lang_feat  # language feature via GloVE
        data_dict['question'] = question['question']  # language stringf
        data_dict["answer"] = answer_label  # answer
        data_dict["answer_str"] = str(self.answer[idx])  # answer
        data_dict["answer_type"] = answer_label_type  # answer category
        data_dict["question_label_type"] = question_label_type  # question category
        data_dict["answer_type_str"] = self.answer_type[idx]  # answer category string
        data_dict["lang_len"] = np.array(lang_len).astype(np.int64)  # length of each description

        # instances
        data_dict['scene_pc'] = point_cloud  # raw point clouds
        data_dict['instance_pc'] = instance_points  # instance point clouds
        data_dict['instance_id'] = instance_indices  # instance labels
        data_dict['instance_descriptor'] = instance_descriptor  # instance descriptor
        data_dict['instance_mask'] = instance_mask  # instance mask
        data_dict['instance_cls'] = instance_semantics  # instance semantics
        data_dict['instance_cls_one_hot'] = instance_semantics_one_hot  # instance semantics
        data_dict['instance_center'] = instance_centers  # instance centers
        # 2d images
        data_dict['img_sequence'] = img_sequence.transpose((0, 3, 1, 2))

        # scene graph
        data_dict['edge_features'] = edge_features  # edge features
        data_dict['edge_labels'] = edge_labels  # edge relationship labels [N, N, CLS]
        data_dict['edge_mask'] = edge_mask  # edge relationship mask [N, N, CLS]
        data_dict['triples'] = triples
        data_dict['pairs'] = pairs
        data_dict['edges'] = edges
        data_dict['predicate_count'] = len(triples)
        # time
        data_dict["load_time"] = time.time() - start

        return data_dict

    @staticmethod
    def collate_fn(batch):
        ans_dict = {}
        lenghth_cat = ['triples', 'pairs', 'edges']
        for key in batch[0].keys():
            if isinstance(batch[0][key], np.ndarray):
                if key in lenghth_cat:
                    ans_dict[key] = [sample[key] for sample in batch]
                else:
                    ans_dict[key] = torch.stack(
                        [torch.from_numpy(sample[key]) for sample in batch],
                        dim=0)
            else:
                ans_dict[key] = [sample[key] for sample in batch]
        ans_dict["answer"] = torch.Tensor(ans_dict["answer"]).long()
        ans_dict["answer_type"] = torch.Tensor(ans_dict["answer_type"]).long()
        ans_dict["question_label_type"] = torch.Tensor(ans_dict["question_label_type"]).long()
        return ans_dict


if __name__ == '__main__':
    with open(CONF.PATH.CLEVER3D_data) as f:
        CLEVER3D = json.load(f)
    train_scene = np.loadtxt(CONF.PATH.R3SCAN_TRAIN, dtype=str).tolist()
    val_scene = np.loadtxt(CONF.PATH.R3SCAN_VAL, dtype=str).tolist()

    val_dset = R3ScanVQAO27R16Dataset(CLEVER3D['questions'], r3scan_scene=val_scene, use_color=True,
                                      use_scene_graph=True)

    for i in range(10000):
        data = val_dset[i]
        print(data["load_time"])
