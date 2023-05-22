"""
Modified from: https://github.com/facebookresearch/votenet/blob/master/scannet/load_scannet_data.py

Load Scannet scenes with vertices and ground truth labels for semantic and instance segmentations
"""

import os
import plyfile
import json
import numpy as np

from sklearn.neighbors import KDTree
from data.R3Scan.r3scan_utils import read_label_mapping

LABEL_MAP_FILE = 'meta_data/3RScan.v2_Mapping.csv'
# OBJ_CLASS_IDS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]) # exclude wall (1), floor (2), ceiling (22)
OBJ_CLASS_IDS = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]) # exclude wall (1), floor (2), ceiling (15)


def read_mesh_vertices_rgb_normal_instanceid(filename):
    '''
    Returns: vertices, rgb, normal, instance_id
    '''
    points = np.loadtxt(filename, skiprows=2)
    return points[:, :3], points[:, 3:6], points[:, 6:9], points[:, 9:10]


def read_semantics(filename):
    '''
    Returns: raw_vertices, raw_semantic (NYU40)
    '''
    plydata = plyfile.PlyData.read(filename)
    point = np.stack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']], 0)
    rgb = np.stack([plydata['vertex']['red'], plydata['vertex']['green'], plydata['vertex']['blue']], 0)
    semantics = plydata['vertex']['RIO27']
    return point.transpose((1, 0)), rgb.transpose((1, 0)), semantics


def read_bbox(filename):
    with open(filename) as f:
        data = json.load(f)

    label_map = read_label_mapping(LABEL_MAP_FILE, label_from='Label', label_to='RIO27_ID')
    boxes = []
    for seg in data['segGroups']:
        axesLengths = seg['obb']['axesLengths']
        centroid = seg['obb']['centroid']
        normalizedAxes = seg['obb']['normalizedAxes']
        object_id = seg['objectId']
        label_id = label_map[seg['label']]
        boxes.append([axesLengths, centroid, normalizedAxes, label_id, object_id])

    return boxes


def judge_obb_intersect(p, obb):
    # judge one point is or not in the obb
    center = np.array(obb["centroid"])
    axis_len = np.array(obb["axesLengths"])
    axis_x = np.array(obb["normalizedAxes"][0:3])
    axis_y = np.array(obb["normalizedAxes"][3:6])
    axis_z = np.array(obb["normalizedAxes"][6:9])
    project_x = axis_x.dot(p - center)
    project_y = axis_y.dot(p - center)
    project_z = axis_z.dot(p - center)
    return -axis_len[0]/2 <= project_x <= axis_len[0]/2 and\
           -axis_len[1]/2 <= project_y <= axis_len[1]/2 and\
           -axis_len[2]/2 <= project_z <= axis_len[2]/2


def export(scene_name):
    scene_name = scene_name.replace('.txt', '')
    output_filename_prefix = os.path.join(OUTPUT_FOLDER, scene_name)
    print('Processing Scene %s...' % output_filename_prefix)

    if os.path.exists(output_filename_prefix + '_vert.npy') and \
        os.path.exists(output_filename_prefix + '_sem_label.npy') and \
        os.path.exists(output_filename_prefix + '_ins_label.npy') and \
        os.path.exists(output_filename_prefix + '_bbox.npy') and \
        os.path.exists(output_filename_prefix + '_aligned_bbox.npy'):
        print('exist!!')
        return

    file_10dim = os.path.join(DATA_PATH, '10dimPoints', scene_name+'.txt')
    label_file = os.path.join(DATA_PATH, '3rscan', scene_name, 'labels.instances.annotated.v2.ply')
    bbox_file = os.path.join(DATA_PATH, '3rscan', scene_name, 'semseg.v2.json')

    vertices, rgb, normal, instance_ids = read_mesh_vertices_rgb_normal_instanceid(file_10dim)
    raw_vertices, raw_rgb, raw_semantics = read_semantics(label_file)
    boxes = read_bbox(bbox_file)

    tree = KDTree(raw_vertices)
    ind = tree.query(vertices, 1)[1]
    semantics = raw_semantics[ind]
    vertices = np.concatenate([vertices, rgb, normal], 1)

    num_instances = len(np.unique(instance_ids))

    aligned_instance_bboxes = np.zeros((num_instances, 9))  # also include object id
    instance_bboxes_tmp = np.zeros((len(boxes), 9))  # also include object id

    for idx, box in enumerate(boxes):
        axesLengths, centroid, normalizedAxes, label_id, object_id = box
        axesLengths = [axesLengths[2], axesLengths[0], axesLengths[1]]
        angle = np.arccos(np.dot(normalizedAxes[0: 3], [0, 1, 0]))

        bbox = np.array(
            [centroid[0], centroid[1], centroid[2],
             axesLengths[0], axesLengths[1], axesLengths[2],
             angle,
             label_id,
             object_id]
        )  # also include object id

        instance_bboxes_tmp[idx, :] = bbox


    for i, obj_id in enumerate(np.unique(instance_ids)):
        obj_id = int(obj_id)
        label_id, count = np.unique(semantics[instance_ids == obj_id], return_counts=True)
        label_id = label_id[count.argmax()]

        label_list = instance_bboxes_tmp[instance_bboxes_tmp[:, -1] == obj_id][:, -2]
        if len(label_list) > 0:
            label_id2 = label_list[0]
        else:
            label_id2 = 0

        if label_id2 != 0:
            label_id = label_id2
            # assert label_id == label_id2, 'label 1 (%d) is not equal to label 2 (%d)' % (label_id, label_id2)
        # bboxes in the original meshes
        obj_pc = vertices[instance_ids[..., 0] == obj_id, 0:3]

        if len(obj_pc) == 0:
            continue

        # Compute axis aligned box
        # An axis aligned bounding box is parameterized by
        # (cx,cy,cz) and (dx,dy,dz) and label id
        # where (cx,cy,cz) is the center point of the box,
        # dx is the x-axis length of the box.
        xmin = np.min(obj_pc[:, 0])
        ymin = np.min(obj_pc[:, 1])
        zmin = np.min(obj_pc[:, 2])
        xmax = np.max(obj_pc[:, 0])
        ymax = np.max(obj_pc[:, 1])
        zmax = np.max(obj_pc[:, 2])

        bbox = np.array(
            [(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2,
             xmax - xmin, ymax - ymin, zmax - zmin,
             0,
             label_id,
             obj_id]
        )  # also include object id

        aligned_instance_bboxes[i, :] = bbox

    instance_bboxes = aligned_instance_bboxes.copy()

    for i in range(instance_bboxes.shape[0]):
        obj_id = instance_bboxes[i, -1]
        if obj_id != 0:
            instance = instance_bboxes_tmp[instance_bboxes_tmp[:, -1] == obj_id]
            if len(instance) > 0:
                instance_bboxes[i, :7] = instance[0, :7]

    if instance_bboxes.shape[0] > 1:
        num_instances = len(np.unique(instance_ids))
        print('Num of instances: ', num_instances)

        bbox_mask = np.in1d(instance_bboxes[:, -2], OBJ_CLASS_IDS) # match the mesh2cap
        instance_bboxes = instance_bboxes[bbox_mask,:]
        aligned_instance_bboxes = aligned_instance_bboxes[bbox_mask,:]
        print('Num of care instances: ', instance_bboxes.shape[0])
    else:
        print("No semantic/instance annotation for test scenes")

    np.save(output_filename_prefix + '_vert.npy', vertices)
    np.save(output_filename_prefix + '_sem_label.npy', semantics)
    np.save(output_filename_prefix + '_ins_label.npy', instance_ids)
    np.save(output_filename_prefix + '_bbox.npy', instance_bboxes)
    np.save(output_filename_prefix + '_aligned_bbox.npy', aligned_instance_bboxes)


if __name__ == '__main__':
    import multiprocessing as mp
    from data.R3Scan.visualize import write_obj, write_bbox, get_3d_box

    RIO27_CLASSES = ['-', 'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'counter',
                     'shelf', 'curtain', 'pillow', 'clothes', 'ceiling', 'fridge', 'tv', 'towel', 'plant', 'box',
                     'nightstand', 'toilet', 'sink', 'lamp', 'bathtub', 'object', 'blanket']

    type2class = {v: k for v, k in enumerate(RIO27_CLASSES)}
    check = False
    OUTPUT_FOLDER = '/218012048/Github/CLEVR3D/data/R3Scan/R3Scan_noSample_27cls/'
    DATA_PATH = '/LiZhen_team/dataset/'

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    if check:
        scene_name = '3b7b33a9-1b11-283e-9b02-e8f35f6ba24c'

        export(scene_name)
        vertices = np.load(OUTPUT_FOLDER + scene_name + '_vert.npy')
        outpath = 'visual_' + scene_name
        os.makedirs(outpath, exist_ok=True)

        write_obj(vertices, os.path.join(outpath, 'scene'), True)

        bbox = np.load(OUTPUT_FOLDER + scene_name + '_bbox.npy')
        bbox_aligned = np.load(OUTPUT_FOLDER + scene_name + '_aligned_bbox.npy')

        for i in range(len(bbox)):
            # box_name = os.path.join(outpath, 'bbox_%d_cls%s.ply' % (i, bbox[i, -2]))
            box_name_align = os.path.join(outpath, 'bbox_%d_%s_align.ply' % (i, type2class[bbox[i, -2]]))

            # write_bbox(get_3d_box(bbox[i, 3:6], 0, bbox[i, 0:3]), 0, box_name)
            write_bbox(get_3d_box(bbox[i, 3:6], 0, bbox[i, 0:3]), 1, box_name_align)

    else:
        p = mp.Pool(processes=mp.cpu_count())
        files = os.listdir(os.path.join(DATA_PATH, '3rscan'))
        print('Total file number: %d' % len(files))
        p.map(export, files)
        p.close()
        p.join()







