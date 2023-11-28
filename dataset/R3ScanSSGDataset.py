import os
import sys
import cv2
import math
import time
import json
import torch
import pickle
import imageio
import numpy as np
import re

from lib.config import CONF
from torch.utils.data import Dataset
from utils import util, op_utils
from transformers import BertTokenizerFast

GLOVE_PICKLE = os.path.join(CONF.PATH.DATA, "glove.p")


def dataset_loading_3RScan(root: str, pth_selection: str, split: str, class_choice: list = None):

    pth_catfile = os.path.join(root, pth_selection, 'classes.txt')
    classNames = util.read_txt_to_list(pth_catfile)

    pth_relationship = os.path.join(root, pth_selection, 'relationships.txt')
    util.check_file_exist(pth_relationship)

    relationNames = util.read_relationships(pth_relationship)

    selected_scans = set()
    data = dict()
    if split == 'train':
        selected_scans = selected_scans.union(util.read_txt_to_list(os.path.join(root, pth_selection, 'train_scans.txt')))
        with open(os.path.join(root, pth_selection, 'relationships_train.json'), "r") as read_file:
            data1 = json.load(read_file)
        data['scans'] = data1['scans']
    elif split == 'val':
        selected_scans = selected_scans.union(util.read_txt_to_list(os.path.join(root, pth_selection, 'validation_scans.txt')))
        with open(os.path.join(root, pth_selection, 'relationships_val.json'), "r") as read_file:
            data1 = json.load(read_file)
        data['scans'] = data1['scans']
    else:
        selected_scans = selected_scans.union(util.read_txt_to_list(os.path.join(root, pth_selection, 'train_scans.txt')))
        with open(os.path.join(root, pth_selection, 'relationships_train.json'), "r") as read_file:
            data1 = json.load(read_file)
        selected_scans = selected_scans.union(util.read_txt_to_list(os.path.join(root, pth_selection, 'validation_scans.txt')))
        with open(os.path.join(root, pth_selection, 'relationships_val.json'), "r") as read_file:
            data2 = json.load(read_file)
        data['scans'] = data1['scans'] + data2['scans']     # merge two json files

    if 'neighbors' in data1:
        data['neighbors'] = data1['neighbors']     #{**data1['neighbors'], **data2['neighbors'], **data3['neighbors']}
    return classNames, relationNames, data, selected_scans


class R3ScanSSGDataset(Dataset):

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
        self.preloading = kwargs.get('preloading', True)
        self.max_img_num = 3
        self.image_dim = (256, 256)
        self.instance_pt_num = num_points
        self.max_instance_num = kwargs.get('max_instance_num', 64)
        self.max_sentence_len = kwargs.get('max_sentence_len', 64)
        self.first_run = kwargs.get('first_run', False)
        self.use_2d = kwargs.get('use_2d', False)
        self.num_ralationships = kwargs.get('num_ralationships', 27)
        text_encoder_type = os.path.join(CONF.PATH.HF, 'bert-base-uncased')
        self.bert_tokenizer = BertTokenizerFast.from_pretrained(text_encoder_type)
        # load data
        self._load_data()

        self.answer_type_dict = {
            'count': 0,
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
        self.question_type_dict = {
            'count': 0,
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
        obj_ids = []

        for data in self.clever3d:
            scene_id = data["scan"]

            if scene_id in self.r3scan_all_scene:
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
                all_lang_questions.append({
                    'scans': scene_id,
                    'question': question,
                    'template_filename': data["template_filename"]
                })
                all_lang_questions_embedding.append(embeddings)
                all_lang_answers.append(answer)
                all_lang_answers_type.append(data["question_type"])
                # obj_ids.append(data["obj_ids"])

        # data_file_name = str(CONF.PATH.CLEVER3D_data).split('/')[-1].replace('.json', '')
        data_dict_name = CONF.PATH.CLEVER3D_answer
        if os.path.exists(data_dict_name):
            print('load answer_dict.json...')
            with open(data_dict_name,) as f:
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
        print('Total %d %s questions in %d scenes...' %
              (len(self.valid_scene_name), self.split, len(set(self.valid_scene_name))))

        return all_lang_questions, all_lang_questions_embedding, all_lang_answers, all_lang_answers_type, obj_ids

    def _load_data(self):
        print("loading %s language data..." % self.split)
        # load language features
        self.question, self.question_emb, self.answer, self.answer_type, self.obj_ids = self._tranform_des()
        self.num_answer_class = len(self.answer)

        print('loading %s scene graph...' % self.split)
        self.scene_graph = json.load(open(os.path.join(CONF.PATH.R3SCAN, "SceneGraphAnnotation.json")))

        self.scene_list = sorted(list(set([d['scans'] for d in self.question if d['scans'] in self.r3scan_all_scene])))

        classNames, relationNames, data, selected_scans = dataset_loading_3RScan(CONF.PATH.BASE, 'data/R3Scan/processed/',
                                                                                 self.split)
        self.relationNames = sorted(relationNames)
        self.classNames = sorted(classNames)
        self.relationship_json, self.objs_json, self.scans, self.nns = self.read_relationship_json(data, selected_scans)

        self.scene_data = {}
        if not self.no_vision and self.preloading:
            print("loading %s point clouds..." % self.split)
            self.scene_data = pickle.load(open(CONF.PATH.H5, 'rb'))

    def read_relationship_json(self, data, selected_scans: list):
        rel = dict()
        objs = dict()
        scans = list()
        nns = None

        if 'neighbors' in data:
            nns = data['neighbors']
        for scan in data['scans']:
            if scan["scan"] == 'fa79392f-7766-2d5c-869a-f5d6cfb62fc6':
                '''
                In the 3RScanV2, the segments on the semseg file and its ply file mismatch. 
                This causes error in loading data.
                To verify this, run check_seg.py
                '''
                continue
            if scan['scan'] not in selected_scans:
                # print('scan not in selected_scans:',scan['scan'])
                continue

            relationships = []
            for realationship in scan["relationships"]:
                relationships.append(realationship)

            objects = {}
            for k, v in scan["objects"].items():
                objects[int(k)] = v

            # filter scans that doesn't have the classes we care
            instances_id = list(objects.keys())
            valid_counter = 0
            for instance_id in instances_id:
                instance_labelName = objects[instance_id]
                if instance_labelName in self.classNames:     # is it a class we care about?
                    valid_counter += 1
                    # break
            if valid_counter < 2:     # need at least two nodes
                continue

            rel[scan["scan"]] = relationships
            scans.append(scan["scan"] + "_" + str(scan["split"]))

            objs[scan["scan"]] = objects

        return rel, objs, scans, nns

    def __getitem__(self, idx):
        start = time.time()

        # get language features
        question = self.question[idx]
        lang_feat = self.question_emb[idx]
        lang_len = len(question['question'].strip('?').split(' '))
        lang_len = lang_len if lang_len <= self.max_sentence_len else self.max_sentence_len
        try:
            answer_label = self.answer_dict[str(self.answer[idx])]
        except:
            answer_label = np.random.randint(len(self.answer_dict))
        answer_label_type = self.answer_type_dict[self.answer_type[idx]]
        question_label_type = self.question_type_dict[self.answer_type[idx]]
        scan_id = question['scans']
        # obj_ids = self.obj_ids[idx]

        ## BERT tokenize
        token_inds = torch.zeros(self.max_sentence_len, dtype=torch.long)
        caption = question['question'].strip('?')
        indices = self.bert_tokenizer.encode(caption, add_special_tokens=True)
        indices = indices[:self.max_sentence_len]
        token_inds[:len(indices)] = torch.tensor(indices)
        token_num = torch.tensor(len(indices), dtype=torch.long)

        img_sequence = np.zeros((self.max_img_num, self.image_dim[0], self.image_dim[1], 3))
        # get images
        if self.use_2d:
            dir_path = os.path.join(CONF.PATH.IMG, question['scans'])
            if os.path.exists(dir_path):
                image_list = [x for x in sorted(os.listdir(dir_path)) if re.search(r'-\d.png', x) != None]
                # image_list = [x for (i, x) in enumerate(image_list) if i % 20 == 0]
                if len(image_list) > 0:
                    imgs = []
                    for img_path in image_list:
                        img = imageio.imread(os.path.join(dir_path, img_path))[..., :3]
                        img = cv2.resize(img, (self.image_dim[0], self.image_dim[1]), interpolation=cv2.INTER_NEAREST)
                        imgs.append(img)
                    imgs = np.stack(imgs, 0)
                    img_num = len(imgs)
                    if img_num > self.max_img_num:
                        img_sequence = imgs[:self.max_img_num]
                    else:
                        img_sequence[:img_num] = imgs

        # get pc
        if self.preloading:
            data = self.scene_data[question['scans']]
            mesh_vertices = data['points']
            instance_labels = data['instances'].reshape(-1, 1)
        else:
            mesh_vertices = np.load(os.path.join('/LiZhen_team/dataset/10dimPoints', question['scans']) + ".npy")
            instance_labels = mesh_vertices[:, -1]
            mesh_vertices = mesh_vertices[:, :-1]

        if not self.use_color:
            point_cloud = mesh_vertices[:, 0:3]     # do not use color for now
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
            point_cloud = self.data_augmentation(point_cloud)
            # # randomly rotate points
            # theta = np.random.rand() * 2 * math.pi
            # point_cloud[:, :3] = np.matmul(
            #     point_cloud[:, :3],
            #     [[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])
            # # randomly dropout
            # point_cloud[:, :3] = random_point_dropout(point_cloud[:, :3])

            # # randomly shuffle
            # point_cloud[:, :3] = shuffle_points(point_cloud[:, :3])

        # from data.R3Scan.visualize import write_obj
        # os.makedirs('visual', exist_ok=True)
        # write_obj(point_cloud, 'visual/pc', rgb=True)

        instance2mask = {}
        instance2mask[0] = 0
        instances_id = list(np.unique(instance_labels))
        instance2labelName = self.objs_json[scan_id]
        ''' 
        Find instances we care abot. Build instance2mask and cat list
        instance2mask maps instances to a mask id. to randomize the order of instance in training.
        '''
        cat = []
        counter = 0
        selected_instances = list(self.objs_json[scan_id].keys())
        filtered_instances = list()
        for i in range(len(instances_id)):
            instance_id = instances_id[i]

            class_id = -1
            if instance_id not in selected_instances:
                instance2mask[instance_id] = 0
                continue
            instance_labelName = instance2labelName[instance_id]
            if instance_labelName in self.classNames:
                class_id = self.classNames.index(instance_labelName)

            if class_id != -1:
                counter += 1
                instance2mask[instance_id] = counter
            else:
                instance2mask[instance_id] = 0

            # mask to cat:
            if (class_id >= 0) and (instance_id > 0):     # insstance 0 is unlabeled.
                filtered_instances.append(instance_id)
                cat.append(class_id)
        ''' random sample points '''
        num_instance = self.max_instance_num
        obj_points = np.zeros([num_instance, self.instance_pt_num, point_cloud.shape[-1]], dtype=np.float32)
        # descriptor = torch.zeros([num_instance, 11])
        gt_class = torch.ones([num_instance], dtype=torch.long) * -1
        instance_mask = np.zeros(num_instance, dtype=np.int64)
        instance_centers = np.zeros((num_instance, 3), dtype=np.float32)

        filtered_instances = filtered_instances[:num_instance]
        for i in range(len(filtered_instances)):
            instance_id = filtered_instances[i]
            obj_pointset = point_cloud[np.where(instance_labels == instance_id)[0], :]
            choice = np.random.choice(len(obj_pointset), self.instance_pt_num, replace=len(obj_pointset) < self.instance_pt_num)
            obj_pointset = obj_pointset[choice, :]
            instance_centers[i] = obj_pointset[:, :3].mean(axis=0)
            # descriptor[i] = op_utils.gen_descriptor(torch.from_numpy(obj_pointset)[:, :3])
            obj_pointset[:, :3] = self.norm_tensor(torch.from_numpy(obj_pointset[:, :3]))
            obj_points[i] = obj_pointset
        ''' Build obj class GT '''
        gt_class[:len(cat)] = torch.from_numpy(np.array(cat))[:self.max_instance_num]
        instance_mask[:len(cat)] = 1

        instance_semantics_one_hot = np.zeros((num_instance, 27), dtype=np.int64)

        edge_labels = np.zeros((self.max_instance_num, self.max_instance_num, self.num_ralationships))
        edge_mask = np.zeros((self.max_instance_num, self.max_instance_num))
        edge_features = np.zeros((self.max_instance_num, self.max_instance_num, 9))
        triples = []
        pairs = []
        edges = []

        if self.use_scene_graph:
            rel_json = self.relationship_json[scan_id]
            # prepare edge feature and labels
            for triple in rel_json:
                if (triple[0] not in filtered_instances) or (triple[1] not in filtered_instances) or (triple[0] == triple[1]):
                    continue

                # indices of triple[0], triple[1] in `objects_id`
                obj1 = filtered_instances.index(triple[0])
                obj2 = filtered_instances.index(triple[1])
                triples.append([obj1, obj2, triple[2]])
                tmp_pair = [obj1, obj2]
                if tmp_pair not in pairs:
                    pairs.append(tmp_pair)

            for triple in triples:
                edge_mask[obj1, obj2] = 1
                edge_labels[obj1, obj2, triple[2]] = 1

            for i in range(self.max_instance_num):
                for j in range(self.max_instance_num):
                    center_offset = instance_centers[i] - instance_centers[j]
                    center_obj = instance_centers[i]
                    center_sub = instance_centers[j]
                    edge_features[i, j] = np.concatenate([center_offset, center_obj, center_sub], 0)

            try:
                s, o = np.split(np.array(pairs), 2, axis=1)     # All have shape (T, 1)
                s, o = [np.squeeze(x, axis=1) for x in [s, o]]     # Now have shape (T,)
                for index, v in enumerate(s):
                    s[index] = v     # s_idx
                for index, v in enumerate(o):
                    o[index] = v     # o_idx
                edges = np.stack((s, o), axis=1)
            except Exception as e:
                edges = []

        data_dict = {}

        # language
        data_dict['scene'] = question['scans']
        data_dict['tokens'] = lang_feat     # language feature via GloVE
        data_dict['question'] = question['question']     # language stringf
        data_dict["answer"] = answer_label     # answer
        data_dict["answer_str"] = str(self.answer[idx])     # answer
        data_dict["answer_type"] = answer_label_type     # answer category
        data_dict["question_label_type"] = question_label_type     # question category
        data_dict["answer_type_str"] = self.answer_type[idx]     # answer category string
        data_dict["lang_len"] = np.array(lang_len).astype(np.int64)     # length of each description
        data_dict['template'] = question['template_filename'].split('.')[0]     # template

        data_dict['token_inds'] = token_inds.numpy().astype(np.int64)
        data_dict['token_num'] = token_num.numpy().astype(np.int64)
        data_dict['context_size'] = np.array(len(cat)).astype(np.int64)

        # instances
        data_dict['instance_pc'] = obj_points     # instance point clouds
        # data_dict['descriptor'] = descriptor
        # data_dict['instance_id'] = filtered_instances  # instance labels
        data_dict['instance_mask'] = instance_mask     # instance mask
        data_dict['instance_cls'] = gt_class     # instance semantics
        data_dict['instance_cls_one_hot'] = instance_semantics_one_hot     # instance semantics
        data_dict['instance_center'] = instance_centers     # instance centers
        # data_dict['instance_usable'] = instance_usable  # instance used in question generation
        # 2d images
        data_dict['img_sequence'] = img_sequence.transpose((0, 3, 1, 2))

        # scene graph
        data_dict['edge_features'] = edge_features     # edge features
        data_dict['edge_labels'] = edge_labels     # edge relationship labels [N, N, CLS]
        data_dict['edge_mask'] = edge_mask     # edge relationship mask [N, N, CLS]
        data_dict['triples'] = triples
        data_dict['pairs'] = pairs
        data_dict['edges'] = edges
        data_dict['predicate_count'] = len(triples)

        # time
        data_dict["load_time"] = time.time() - start

        # for key in data_dict.keys():
        #     if isinstance(data_dict[key], np.ndarray):
        #         print(key, data_dict[key].dtype)
        # sys.exit()

        return data_dict

    def data_augmentation(self, points):
        # random rotate
        matrix = np.eye(3)
        matrix[0:3, 0:3] = op_utils.rotation_matrix([0, 0, 1], np.random.uniform(0, 2 * np.pi, 1))
        centroid = points[:, :3].mean(0)
        points[:, :3] -= centroid
        points[:, :3] = np.dot(points[:, :3], matrix.T)
        if self.use_normal:
            ofset = 3
            if self.use_rgb:
                ofset += 3
            points[:, ofset:3 + ofset] = np.dot(points[:, ofset:3 + ofset], matrix.T)

        ## Add noise
        # ## points
        # noise = np.random.normal(0,1e-3,[points.shape[0],3]) # 1 mm std
        # points[:,:3] += noise

        # ## colors
        # if self.use_rgb:
        #     noise = np.random.normal(0,0.078,[points.shape[0],3])
        #     colors = points[:,3:6]
        #     colors += noise
        #     colors[np.where(colors>1)] = 1
        #     colors[np.where(colors<-1)] = -1

        # ## normals
        # if self.use_normal:
        #     ofset=3
        #     if self.use_rgb:
        #         ofset+=3
        #     normals = points[:,ofset:3+ofset]
        #     normals = np.dot(normals, matrix.T)

        #     noise = np.random.normal(0,1e-4,[points.shape[0],3])
        #     normals += noise
        #     normals = normals/ np.linalg.norm(normals)
        return points

    def norm_tensor(self, points):
        assert points.ndim == 2
        assert points.shape[1] == 3
        centroid = torch.mean(points, dim=0)     # N, 3
        points -= centroid     # n, 3, npts
        furthest_distance = points.pow(2).sum(1).sqrt().max()     # find maximum distance for each n -> [n]
        points /= furthest_distance
        return points

    @staticmethod
    def collate_fn(batch):
        ans_dict = {}
        lenghth_cat = ['triples', 'pairs', 'edges']
        for key in batch[0].keys():
            if isinstance(batch[0][key], np.ndarray):
                if key in lenghth_cat:
                    ans_dict[key] = [sample[key] for sample in batch]
                else:
                    ans_dict[key] = torch.stack([torch.from_numpy(sample[key]) for sample in batch], dim=0)
            elif isinstance(batch[0][key], torch.Tensor):
                ans_dict[key] = torch.stack([sample[key] for sample in batch], dim=0)
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

    # train_dset = R3ScanVQADataset(CLEVER3D['questions'], r3scan_scene=train_scene, use_color=True)
    val_dset = R3ScanSSGDataset(CLEVER3D['questions'],
                                r3scan_scene=val_scene,
                                use_color=True,
                                use_scene_graph=True,
                                preloading=True)
    # (val_dset[0])
    # (train_dset[1352])
    for i in range(10000):
        data = val_dset[i]
        print(data["load_time"])
