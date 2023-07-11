if __name__ == '__main__' and __package__ is None:
    from os import sys

    sys.path.append('../')
import h5py, os
import trimesh
from utils import util_ply
import numpy as np
from tqdm import tqdm

def load_mesh(path, label_file, use_rgb, use_normal):
    result = dict()
    if label_file == 'labels.instances.align.annotated.v2.ply' or label_file == 'labels.instances.align.annotated.ply':
        if use_rgb:
            plydata = util_ply.load_rgb(path)
        else:
            plydata = trimesh.load(os.path.join(path, label_file), process=False)

        points = np.array(plydata.vertices)
        instances = util_ply.read_labels(plydata).flatten()

        if use_rgb:
            r = plydata.metadata['ply_raw']['vertex']['data']['red']
            g = plydata.metadata['ply_raw']['vertex']['data']['green']
            b = plydata.metadata['ply_raw']['vertex']['data']['blue']
            rgb = np.stack([r, g, b]).squeeze().transpose()
            points = np.concatenate((points, rgb), axis=1)
        if use_normal:
            nx = plydata.metadata['ply_raw']['vertex']['data']['nx']
            ny = plydata.metadata['ply_raw']['vertex']['data']['ny']
            nz = plydata.metadata['ply_raw']['vertex']['data']['nz']
            normal = np.stack([nx, ny, nz]).squeeze().transpose()
            points = np.concatenate((points, normal), axis=1)

        result['points'] = points
        result['instances'] = instances

    return result


scene_list = read_txt_to_list(os.path.join(define.FILE_PATH, 'all_scans.txt'))

f = h5py.File(define.H5_PATH, "w")

for name in tqdm(scene_list):
    path = os.path.join(define.DATA_PATH, name)
    data = load_mesh(path, "labels.instances.align.annotated.v2.ply", True, True)
    points = data['points']
    instances = data['instances']
    g = f.create_group(name)
    g.create_dataset('points', data=points)
    g.create_dataset('instances', data=instances)

f = h5py.File(define.H5_PATH, "r")
print(f[name]['points'].shape, f[name]['instances'].shape)
