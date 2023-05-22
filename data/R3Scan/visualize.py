#!/usr/bin/env python
# encoding: utf-8
'''
@author: Xu Yan
@file: visualize.py
@time: 2021/10/18 14:18
'''
import numpy as np
from utils.box_util import get_3d_box



def write_ply(verts, colors, indices, output_file):
    if colors is None:
        colors = np.zeros_like(verts)
    if indices is None:
        indices = []

    file = open(output_file, 'w')
    file.write('ply \n')
    file.write('format ascii 1.0\n')
    file.write('element vertex {:d}\n'.format(len(verts)))
    file.write('property float x\n')
    file.write('property float y\n')
    file.write('property float z\n')
    file.write('property uchar red\n')
    file.write('property uchar green\n')
    file.write('property uchar blue\n')
    file.write('element face {:d}\n'.format(len(indices)))
    file.write('property list uchar uint vertex_indices\n')
    file.write('end_header\n')
    for vert, color in zip(verts, colors):
        file.write("{:f} {:f} {:f} {:d} {:d} {:d}\n".format(vert[0], vert[1], vert[2], int(color[0] * 255),
                                                            int(color[1] * 255), int(color[2] * 255)))
    for ind in indices:
        file.write('3 {:d} {:d} {:d}\n'.format(ind[0], ind[1], ind[2]))
    file.close()


def write_bbox(corners, mode, output_file):
    """
    bbox: (cx, cy, cz, lx, ly, lz, r), center and length in three axis, the last is the rotation
    output_file: string
    """

    def create_cylinder_mesh(radius, p0, p1, stacks=10, slices=10):

        import math

        def compute_length_vec3(vec3):
            return math.sqrt(vec3[0] * vec3[0] + vec3[1] * vec3[1] + vec3[2] * vec3[2])

        def rotation(axis, angle):
            rot = np.eye(4)
            c = np.cos(-angle)
            s = np.sin(-angle)
            t = 1.0 - c
            axis /= compute_length_vec3(axis)
            x = axis[0]
            y = axis[1]
            z = axis[2]
            rot[0, 0] = 1 + t * (x * x - 1)
            rot[0, 1] = z * s + t * x * y
            rot[0, 2] = -y * s + t * x * z
            rot[1, 0] = -z * s + t * x * y
            rot[1, 1] = 1 + t * (y * y - 1)
            rot[1, 2] = x * s + t * y * z
            rot[2, 0] = y * s + t * x * z
            rot[2, 1] = -x * s + t * y * z
            rot[2, 2] = 1 + t * (z * z - 1)
            return rot

        verts = []
        indices = []
        diff = (p1 - p0).astype(np.float32)
        height = compute_length_vec3(diff)
        for i in range(stacks + 1):
            for i2 in range(slices):
                theta = i2 * 2.0 * math.pi / slices
                pos = np.array([radius * math.cos(theta), radius * math.sin(theta), height * i / stacks])
                verts.append(pos)
        for i in range(stacks):
            for i2 in range(slices):
                i2p1 = math.fmod(i2 + 1, slices)
                indices.append(np.array([(i + 1) * slices + i2, i * slices + i2, i * slices + i2p1], dtype=np.uint32))
                indices.append(
                    np.array([(i + 1) * slices + i2, i * slices + i2p1, (i + 1) * slices + i2p1], dtype=np.uint32))
        transform = np.eye(4)
        va = np.array([0, 0, 1], dtype=np.float32)
        vb = diff
        vb /= compute_length_vec3(vb)
        axis = np.cross(vb, va)
        angle = np.arccos(np.clip(np.dot(va, vb), -1, 1))
        if angle != 0:
            if compute_length_vec3(axis) == 0:
                dotx = va[0]
                if (math.fabs(dotx) != 1.0):
                    axis = np.array([1, 0, 0]) - dotx * va
                else:
                    axis = np.array([0, 1, 0]) - va[1] * va
                axis /= compute_length_vec3(axis)
            transform = rotation(axis, -angle)
        transform[:3, 3] += p0
        verts = [np.dot(transform, np.array([v[0], v[1], v[2], 1.0])) for v in verts]
        verts = [np.array([v[0], v[1], v[2]]) / v[3] for v in verts]

        return verts, indices

    def get_bbox_edges(bbox_min, bbox_max):
        def get_bbox_verts(bbox_min, bbox_max):
            verts = [
                np.array([bbox_min[0], bbox_min[1], bbox_min[2]]),
                np.array([bbox_max[0], bbox_min[1], bbox_min[2]]),
                np.array([bbox_max[0], bbox_max[1], bbox_min[2]]),
                np.array([bbox_min[0], bbox_max[1], bbox_min[2]]),

                np.array([bbox_min[0], bbox_min[1], bbox_max[2]]),
                np.array([bbox_max[0], bbox_min[1], bbox_max[2]]),
                np.array([bbox_max[0], bbox_max[1], bbox_max[2]]),
                np.array([bbox_min[0], bbox_max[1], bbox_max[2]])
            ]
            return verts

        box_verts = get_bbox_verts(bbox_min, bbox_max)
        edges = [
            (box_verts[0], box_verts[1]),
            (box_verts[1], box_verts[2]),
            (box_verts[2], box_verts[3]),
            (box_verts[3], box_verts[0]),

            (box_verts[4], box_verts[5]),
            (box_verts[5], box_verts[6]),
            (box_verts[6], box_verts[7]),
            (box_verts[7], box_verts[4]),

            (box_verts[0], box_verts[4]),
            (box_verts[1], box_verts[5]),
            (box_verts[2], box_verts[6]),
            (box_verts[3], box_verts[7])
        ]
        return edges

    radius = 0.03
    offset = [0, 0, 0]
    verts = []
    indices = []
    colors = []

    box_min = np.min(corners, axis=0)
    box_max = np.max(corners, axis=0)
    palette = {
        0: [0, 255, 0],  # gt
        1: [0, 0, 255]  # pred
    }
    chosen_color = palette[mode]
    edges = get_bbox_edges(box_min, box_max)
    for k in range(len(edges)):
        cyl_verts, cyl_ind = create_cylinder_mesh(radius, edges[k][0], edges[k][1])
        cur_num_verts = len(verts)
        cyl_color = [[c / 255 for c in chosen_color] for _ in cyl_verts]
        cyl_verts = [x + offset for x in cyl_verts]
        cyl_ind = [x + cur_num_verts for x in cyl_ind]
        verts.extend(cyl_verts)
        indices.extend(cyl_ind)
        colors.extend(cyl_color)

    write_ply(verts, colors, indices, output_file)


def write_obj(points, file, rgb=False):
    fout = open('%s.obj' % file, 'w')
    for i in range(points.shape[0]):
        if not rgb:
            fout.write('v %f %f %f %d %d %d\n' % (
                points[i, 0], points[i, 1], points[i, 2], 255, 255, 0))
        else:
            fout.write('v %f %f %f %d %d %d\n' % (
                points[i, 0], points[i, 1], points[i, 2], points[i, -3] * 255, points[i, -2] * 255,
                points[i, -1] * 255))


if __name__ == '__main__':
    import os
    from sklearn.neighbors import KDTree

    from data.scan3r.load_scan3r_data import read_mesh_vertices_rgb_normal_instanceid, read_semantics, read_bbox
    scene_name = '0988ea72-eb32-2e61-8344-99e2283c2728'

    outpath = 'visual_' + scene_name
    os.makedirs(outpath, exist_ok=True)

    data_path = '/data/dataset/'

    file_10dim = os.path.join(data_path, '10dimPoints', scene_name+'.txt')
    label_file = os.path.join(data_path, '3rscan', scene_name, 'labels.instances.annotated.v2.ply')
    bbox_file = os.path.join(data_path, '3rscan', scene_name, 'semseg.v2.json')

    vertices, rgb, normal, instance_ids = read_mesh_vertices_rgb_normal_instanceid(file_10dim)
    raw_vertices, raw_rgb, raw_semantics = read_semantics(label_file)
    boxes = read_bbox(bbox_file)

    tree = KDTree(raw_vertices)
    ind = tree.query(vertices, 1)[1]
    semantics = raw_semantics[ind]

    write_obj(np.concatenate([vertices, rgb/255], 1), os.path.join(outpath, 'scene'), True)

    print(len(boxes))
    print(len(np.unique(instance_ids)))

    num_instances = len(np.unique(instance_ids))

    aligned_instance_bboxes = np.zeros((num_instances, 9))  # also include object id
    instance_bboxes_tmp = np.zeros((num_instances, 9))  # also include object id

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

    print(instance_bboxes_tmp)
    print(instance_bboxes_tmp.shape)

    for i, obj_id in enumerate(np.unique(instance_ids)):
        obj_id = int(obj_id)
        print(obj_id)
        label_id, count = np.unique(semantics[instance_ids == obj_id], return_counts=True)
        print(label_id, count)
        label_id = label_id[count.argmax()]
        print(label_id)

        label_list = instance_bboxes_tmp[instance_bboxes_tmp[:, -1] == obj_id][:, -2]
        if len(label_list) > 0:
            label_id2 = label_list[0]
        else:
            label_id2 = 0
        print(label_id2)
        assert label_id == label_id2
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

    print(np.unique(instance_bboxes_tmp[:, -1]))
    for i in range(instance_bboxes.shape[0]):
        obj_id = instance_bboxes[i, -1]
        if obj_id != 0:
            print(obj_id)
            instance = instance_bboxes_tmp[instance_bboxes_tmp[:, -1] == obj_id]
            if len(instance) > 0:
                instance_bboxes[i, :7] = instance[0, :7]


    for i in range(len(instance_bboxes)):
        box_name = os.path.join(outpath, 'bbox_%d.ply' % i)
        box_name_align = os.path.join(outpath, 'bbox_%d_align.ply' % i)

        write_bbox(get_3d_box(instance_bboxes[i, 3:6], 0, instance_bboxes[i, 0:3]), 0, box_name)
        write_bbox(get_3d_box(aligned_instance_bboxes[i, 3:6], 0, aligned_instance_bboxes[i, 0:3]), 1, box_name_align)






