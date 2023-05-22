import torch
import math
import numpy as np
from pytorch_lightning.metrics import Metric
from torchmetrics.functional import recall, f1


class ClfAccuracy(Metric):
    def __init__(self, ignore_labels, dist_sync_on_step=False, compute_on_step=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step, compute_on_step=compute_on_step)

        self.ignore_labels = ignore_labels
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, predictions, gt_labels) -> None:
        """ Get the prediction statistics: accuracy, correctly/wrongly predicted test examples
        :param logits: The output of the model (predictions) of size: B x N_Objects x N_Classes
        :param gt_labels: The ground truth labels of size: B x N_Objects
        :param ignore_label: The label of the padding class (to be ignored)
        :return: The mean accuracy and lists of correct and wrong predictions
        """
        valid_indices = gt_labels != self.ignore_labels

        predictions = predictions[valid_indices]
        gt_labels = gt_labels[valid_indices]

        correct_guessed = gt_labels == predictions
        assert (type(correct_guessed) == torch.Tensor)

        self.correct += correct_guessed.sum()
        self.total += gt_labels.numel()

    def compute(self):
        return self.correct.float() / self.total


class ClassAccuracy(Metric):
    def __init__(self, dist_sync_on_step=False, compute_on_step=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step, compute_on_step=compute_on_step)

        self.ref_dict = {'count': 0,
                         'equal_color': 0,
                         'equal_integer': 0,
                         'equal_material': 0,
                         'equal_shape': 0,
                         'equal_size': 0,
                         'exist': 0,
                         'greater_than': 0,
                         'less_than': 0,
                         'query_color': 0,
                         'query_material': 0,
                         'query_shape': 0,
                         'query_size': 0,
                         'query_label': 0}

        self.val_ref_acc_num = self.ref_dict.copy()
        self.val_ref_total = self.ref_dict.copy()

    def update(self, predictions, gt_labels, answer_type) -> None:
        """ Get the prediction statistics: accuracy, correctly/wrongly predicted test examples
        :param logits: The output of the model (predictions) of size: B x N_Objects x N_Classes
        :param gt_labels: The ground truth labels of size: B x N_Objects
        :param ignore_label: The label of the padding class (to be ignored)
        :return: The mean accuracy and lists of correct and wrong predictions
        """
        batch_size = predictions.shape[0]
        for b in range(batch_size):
            if (predictions[b] == gt_labels[b]).cpu().item():
                self.val_ref_acc_num[answer_type[b]] += 1
            self.val_ref_total[answer_type[b]] += 1

    def compute(self):
        results_dict = {}
        for k in list(self.val_ref_acc_num.keys()):
            results_dict[k] = [self.val_ref_acc_num[k] / (self.val_ref_total[k] + 1e-6),
                               self.val_ref_acc_num[k],
                               self.val_ref_total[k]
                               ]

        self.val_ref_acc_num = self.ref_dict.copy()
        self.val_ref_total = self.ref_dict.copy()

        return results_dict


class SceneGraphEval(Metric):
    def __init__(self, dist_sync_on_step=False, compute_on_step=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step, compute_on_step=compute_on_step)
        self.top5_ratio_o = []
        self.top10_ratio_o = []
        self.top3_ratio_r = []
        self.top5_ratio_r = []
        self.top50_predicate = []
        self.top100_predicate = []
        self.pred_relations_ret = []
        self.scene = []

    def update(self, data_dict) -> None:
        batch_size = data_dict['objects_cls_pred'].shape[0]

        for i in range(batch_size):
            if data_dict['scene'] not in self.scene:
                object_pred = data_dict['objects_cls_pred'][i].permute(1, 0).detach().cpu()
                object_cat = data_dict['instance_cls'][i].detach().cpu()
                # here, need to notice the relationships between keys 'edges', 'pairs' and 'triples'
                # 'pairs' is the none duplicate list of the first two column of 'triples', while 'edges' corresponds to the index
                # the 'predicate_predict' takes the mapping relation with 'pairs'
                predicate_count = data_dict["predicate_count"][i]
                predicate_pred = data_dict["predicate_predict"][i].detach().cpu()
                predicate_label = data_dict["edge_labels"][i].long().detach().cpu()
                pairs = np.array(data_dict["pairs"][i]).reshape((-1, 2))
                edge_mask = data_dict['edge_mask'][0].bool().detach().cpu().numpy()
                try:
                    edges = np.array(data_dict["edges"][i]).reshape((-1, 2))
                except:
                    continue
                triples = np.array(data_dict["triples"][i]).reshape((-1, 3))

                predicate_cat = triples[:, 2]
                predicate_pred = predicate_pred.numpy()
                predicate_pred_expand = np.zeros([triples.shape[0], predicate_pred.shape[2]])
                for index, triple in enumerate(triples):
                    predicate_pred_expand[index] = predicate_pred[triple[[0]], triple[[1]]]
                predicate_pred_expand = torch.Tensor(predicate_pred_expand)
                predicate_cat = torch.Tensor(predicate_cat).long()

                self.top5_ratio_o.append(
                    recall(object_pred[object_cat != -1],
                           object_cat[object_cat != -1],
                           top_k=5,
                           num_classes=160))
                self.top10_ratio_o.append(
                    recall(object_pred[object_cat != -1],
                           object_cat[object_cat != -1],
                           top_k=10,
                           num_classes=160))
                self.top3_ratio_r.append(f1(torch.nn.functional.sigmoid(predicate_pred_expand),
                                            predicate_cat,
                                            top_k=3,
                                            num_classes=7,
                                            average='macro',
                                            multiclass=True
                                            ))
                self.top5_ratio_r.append(f1(torch.nn.functional.sigmoid(predicate_pred_expand),
                                            predicate_cat,
                                            top_k=5,
                                            num_classes=7,
                                            average='macro',
                                            multiclass=True))
                # self.top3_ratio_r.append(f1(torch.nn.functional.sigmoid(predicate_pred[edge_mask]),
                #                             predicate_label[edge_mask],
                #                             top_k=3,
                #                             num_classes=16))
                # self.top5_ratio_r.append(f1(torch.nn.functional.sigmoid(predicate_pred[edge_mask]),
                #                             predicate_label[edge_mask],
                #                             top_k=5,
                #                             num_classes=16))

                # store the index
                object_pred = object_pred.numpy()
                object_logits = np.max(object_pred, axis=1)
                object_idx = np.argmax(object_pred, axis=1)
                obj_scores_per_rel = object_logits[edges].prod(1)
                overall_scores = obj_scores_per_rel[:, None] * predicate_pred_expand.numpy()
                score_inds = argsort_desc(overall_scores)[:100]
                pred_rels = np.column_stack((pairs[score_inds[:, 0]], score_inds[:, 1]))

                self.top50_predicate.append(topk_triplet(pred_rels, triples, 50))
                self.top100_predicate.append(topk_triplet(pred_rels, triples, 100))
                self.scene.append(data_dict['scene'])
                # self.pred_relations_ret.append(pred_rels)

    def compute(self):
        metric_dict = {
            'Recall@5_object': np.mean(self.top5_ratio_o),
            'Recall@10_object': np.mean(self.top10_ratio_o),
            'F1@3_predicate': np.mean(self.top3_ratio_r),
            'F1@5_predicate': np.mean(self.top5_ratio_r),
            'Recall@50_relationship': np.mean(self.top50_predicate),
            'Recall@100_relationship': np.mean(self.top100_predicate),
            # 'pred_relations_ret': np.mean(self.pred_relations_ret),
            'TotalScene': len(self.scene),
        }
        self.scene = []
        return metric_dict


class SceneGraphEvalSSG(Metric):
    def __init__(self, dist_sync_on_step=False, compute_on_step=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step, compute_on_step=compute_on_step)
        self.top5_ratio_o = []
        self.top10_ratio_o = []
        self.top3_ratio_r = []
        self.top5_ratio_r = []
        self.top50_predicate = []
        self.top100_predicate = []
        self.pred_relations_ret = []
        self.scene = []

    def update(self, data_dict) -> None:
        batch_size = data_dict['objects_cls_pred'].shape[0]

        for i in range(batch_size):
            if data_dict['scene'] not in self.scene:
                object_pred = torch.nn.functional.sigmoid(data_dict['objects_cls_pred'][i]).permute(1, 0).detach().cpu()
                object_cat = data_dict['instance_cls'][i].detach().cpu()
                # here, need to notice the relationships between keys 'edges', 'pairs' and 'triples'
                # 'pairs' is the none duplicate list of the first two column of 'triples', while 'edges' corresponds to the index
                # the 'predicate_predict' takes the mapping relation with 'pairs'
                predicate_count = data_dict["predicate_count"][i]
                predicate_pred = data_dict["predicate_predict"][i].detach().cpu()
                predicate_label = data_dict["edge_labels"][i].long().detach().cpu()
                pairs = np.array(data_dict["pairs"][i]).reshape((-1, 2))
                edge_mask = data_dict['edge_mask'][0].bool().detach().cpu().numpy()
                try:
                    edges = np.array(data_dict["edges"][i]).reshape((-1, 2))
                except:
                    continue
                triples = np.array(data_dict["triples"][i]).reshape((-1, 3))

                predicate_cat = triples[:, 2]
                predicate_pred_expand = torch.nn.functional.sigmoid(data_dict['predicate_predict'][i]).cpu()
                predicate_cat = torch.Tensor(predicate_cat).long()

                self.top5_ratio_o.append(
                    recall(object_pred[object_cat != -1],
                           object_cat[object_cat != -1],
                           top_k=5,
                           num_classes=160))

                self.top10_ratio_o.append(
                    recall(object_pred[object_cat != -1],
                           object_cat[object_cat != -1],
                           top_k=10,
                           num_classes=160))
                self.top3_ratio_r.append(
                    recall(predicate_pred_expand,
                           predicate_cat,
                           top_k=3,
                           num_classes=7)
                )
                self.top5_ratio_r.append(
                    recall(predicate_pred_expand,
                           predicate_cat,
                           top_k=5,
                           num_classes=7)
                )
                # self.top3_ratio_r.append(f1(torch.nn.functional.sigmoid(predicate_pred[edge_mask]),
                #                             predicate_label[edge_mask],
                #                             top_k=3,
                #                             num_classes=16))
                # self.top5_ratio_r.append(f1(torch.nn.functional.sigmoid(predicate_pred[edge_mask]),
                #                             predicate_label[edge_mask],
                #                             top_k=5,
                #                             num_classes=16))

                # store the index
                object_pred = object_pred.numpy()
                object_logits = np.max(object_pred, axis=1)
                object_idx = np.argmax(object_pred, axis=1)
                obj_scores_per_rel = object_logits[edges].prod(1)
                overall_scores = obj_scores_per_rel[:, None] * predicate_pred_expand.numpy()
                score_inds = argsort_desc(overall_scores)[:100]
                pred_rels = np.column_stack((pairs[score_inds[:, 0]], score_inds[:, 1]))

                self.top50_predicate.append(topk_triplet(pred_rels, triples, 50))
                self.top100_predicate.append(topk_triplet(pred_rels, triples, 100))
                self.scene.append(data_dict['scene'])
                # self.pred_relations_ret.append(pred_rels)

    def compute(self):
        metric_dict = {
            'Recall@5_object': np.mean(self.top5_ratio_o),
            'Recall@10_object': np.mean(self.top10_ratio_o),
            'Recall@3_predicate': np.mean(self.top3_ratio_r),
            'Recall@5_predicate': np.mean(self.top5_ratio_r),
            'Recall@50_relationship': np.mean(self.top50_predicate),
            'Recall@100_relationship': np.mean(self.top100_predicate),
            # 'pred_relations_ret': np.mean(self.pred_relations_ret),
            'TotalScene': len(self.scene),
        }
        self.scene = []
        return metric_dict


class SceneGraphEvalSSG2(Metric):
    def __init__(self, dist_sync_on_step=False, compute_on_step=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step, compute_on_step=compute_on_step)
        self.top5_ratio_o = []
        self.top10_ratio_o = []
        self.top3_ratio_r = []
        self.top5_ratio_r = []
        self.topk_predicate = []
        self.pred_relations_ret = []
        self.scene = []

    def update(self, data_dict) -> None:
        batch_size = data_dict['objects_cls_pred'].shape[0]

        for i in range(batch_size):
            if data_dict['scene'][i] not in self.scene and len(data_dict['edges'][i]) > 1:
                object_pred = torch.nn.functional.sigmoid(data_dict['objects_cls_pred'][i]).permute(1, 0).detach().cpu()
                object_cat = data_dict['instance_cls'][i].detach().cpu()

                triples = np.array(data_dict["triples"][i]).reshape((-1, 3))

                predicate_cat = triples[:, 2]
                predicate_pred_expand = torch.nn.functional.sigmoid(data_dict['predicate_predict'][i]).cpu()
                predicate_cat = torch.Tensor(predicate_cat).long()

                self.top5_ratio_o.append(
                    recall(object_pred[object_cat != -1],
                           object_cat[object_cat != -1],
                           top_k=5,
                           num_classes=27))

                self.top10_ratio_o.append(
                    recall(object_pred[object_cat != -1],
                           object_cat[object_cat != -1],
                           top_k=10,
                           num_classes=27))
                self.top3_ratio_r.append(
                    f1(predicate_pred_expand,
                       predicate_cat,
                       top_k=3,
                       num_classes=16, threshold=0.1)
                )
                self.top5_ratio_r.append(
                    f1(predicate_pred_expand,
                       predicate_cat,
                       top_k=5,
                       num_classes=16, threshold=0.1)
                )

                s_gt = object_pred.argmax(-1)[triples[:, 0]]
                t_gt = object_pred.argmax(-1)[triples[:, 1]]
                gt_rel = torch.stack([s_gt, t_gt, torch.Tensor(triples[:, -1])], 1).long()
                self.topk_predicate += evaluate_topk(gt_rel,
                                                     object_pred[object_cat != -1],
                                                     predicate_pred_expand,
                                                     triples[:, :2],
                                                     multi_rel_outputs=False,
                                                     threshold=0.5,
                                                     k=100)


    def compute(self):
        metric_dict = {
            'Recall@5_object': np.mean(self.top5_ratio_o),
            'Recall@10_object': np.mean(self.top10_ratio_o),
            'Recall@3_predicate': np.mean(self.top3_ratio_r),
            'Recall@5_predicate': np.mean(self.top5_ratio_r),
            # 'pred_relations_ret': np.mean(self.pred_relations_ret),
            'TotalScene': len(self.scene),
        }
        ntop_k = np.asarray(self.topk_predicate)
        ks = set([1, 2, 3, 5, 10, 50, 100])
        for i in [0, 0.05, 0.1, 0.2, 0.5, 0.9]:
            ks.add(int(math.ceil(100 * i + 1e-9)))
        for k in sorted(ks):
            R = (ntop_k <= k).sum() / len(ntop_k)
            metric_dict['Recall@%d_relationship' % k] = R

        self.scene = []

        return metric_dict


def get_gt(objs_target, rels_target, edges, instance2mask, multi_rel_outputs):
    gt_edges = []  # initialize
    idx2instance = torch.zeros_like(objs_target)
    for key, value in instance2mask.items():
        if value > 0:
            idx2instance[value - 1] = key

    for edge_index in range(len(edges)):
        idx_eo = edges[edge_index][0].cpu().numpy().item()
        idx_os = edges[edge_index][1].cpu().numpy().item()

        target_eo = objs_target[idx_eo].cpu().numpy().item()
        target_os = objs_target[idx_os].cpu().numpy().item()
        target_rel = []  # there might be multiple
        if multi_rel_outputs:
            assert rels_target.ndim == 2
            for i in range(rels_target.size(1)):
                if rels_target[edge_index, i] == 1:
                    target_rel.append(i)
        else:
            assert rels_target.ndim == 1
            if rels_target[edge_index] > 0:
                target_rel.append(rels_target[edge_index].cpu().numpy().item())
        gt_edges.append([target_eo, target_os, target_rel,
                         idx2instance[idx_eo], idx2instance[idx_os]])
    return gt_edges


def evaluate_topk(gt_rel, objs_pred, rels_pred, edges, multi_rel_outputs, threshold=0.5, k=40):
    top_k = list()
    objs_pred = np.exp(objs_pred.detach().cpu())
    if multi_rel_outputs:
        # sigmoids
        rels_pred = rels_pred.detach().cpu()
    else:
        # log_softmax
        rels_pred = np.exp(rels_pred.detach().cpu())

    for edge in range(len(edges)):

        edge_from = edges[edge][0]
        edge_to = edges[edge][1]
        rel_predictions = rels_pred[edge]
        objs_pred_1 = objs_pred[edge_from]
        objs_pred_2 = objs_pred[edge_to]
        node_score = torch.einsum('n,m->nm', objs_pred_1, objs_pred_2)
        conf_matrix = torch.einsum('nl,m->nlm', node_score, rel_predictions)
        conf_matrix_1d = conf_matrix.reshape(-1)
        sorted_conf_matrix, sorted_args_1d = torch.sort(conf_matrix_1d, descending=True)  # 1D
        if k < 1:
            maxk = len(sorted_conf_matrix)
            maxk = min(len(sorted_conf_matrix), maxk)
        else:
            maxk = k
        sorted_conf_matrix = sorted_conf_matrix[:maxk]
        sorted_args_1d = sorted_args_1d[:maxk]
        e = gt_rel[edge]
        gt_s = e[0]
        gt_t = e[1]
        gt_r = e[2]
        temp_topk = []

        if isinstance(gt_r, np.int64):
            # Ground truth is None
            indices = torch.where(sorted_conf_matrix < threshold)[0]
            if len(indices) == 0:
                index = maxk + 1
            else:
                index = sorted(indices)[0].item() + 1
            temp_topk.append(index)

        predicate = gt_r
        try:
            gt_conf = conf_matrix[gt_s, gt_t, predicate]
        except:
            import pdb; pdb.set_trace()
            print('conf_matrix', conf_matrix.shape)
            print('objs_pred', objs_pred.shape)
            print('rels_pred', rels_pred.shape)
            print('edges', edges.shape)
            print('gt_s', gt_s.shape)
            print('gt_t', gt_t.shape)
        indices = torch.where(sorted_conf_matrix == gt_conf)[0]
        if len(indices) == 0:
            index = maxk + 1
        else:
            index = sorted(indices)[0].item() + 1
        temp_topk.append(index)
        temp_topk = sorted(temp_topk)
        top_k += temp_topk
    return top_k


def topk_ratio(logits, category, k):
    """
    Parameters
    ----------
    logits: [N C] N objects/relationships with C categroy
    category: [N 1] N objects/relationships
    k:  top k

    Returns topk_ratio: recall of top k (R@k)
    -------
    """
    topk_pred = np.argsort(-logits, axis=1)[:, :k]  # descending order
    topk_ratio = 0
    for index, x in enumerate(topk_pred):
        if category[index] in x:
            topk_ratio += 1
    topk_ratio /= category.shape[0]
    return topk_ratio


def topk_triplet(pred_tri, gt_tri, k):
    """
    Parameters
    ----------
    pred_tri: multiplying predict scores results
    gt_tri: triplets exist in the scene
    k:  top k

    Returns ratio: recall of top k (R@k)
    -------
    """
    # assert len(tri)>=k
    ratio = 0
    gt = gt_tri.tolist()
    pred = pred_tri[:k]
    for item in pred:
        line = item.tolist()
        if line in gt:
            ratio += 1
    ratio /= len(gt_tri)
    return ratio


def argsort_desc(scores):
    """
    Returns the indices that sort scores descending in a smart way
    :param scores: Numpy array of arbitrary size
    :return: an array of size [numel(scores), dim(scores)] where each row is the index you'd
             need to get the score.
    """
    return np.column_stack(np.unravel_index(np.argsort(-scores.ravel()), scores.shape))
