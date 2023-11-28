import os
import sys
from easydict import EasyDict

CONF = EasyDict()

# path
CONF.PATH = EasyDict()
CONF.PATH.BASE = os.getcwd()
CONF.PATH.DATA = os.path.join("/221019046/", "Data")
CONF.PATH.R3SCAN = os.path.join(CONF.PATH.BASE, "data/R3Scan")
CONF.PATH.LIB = os.path.join(CONF.PATH.BASE, "lib")
CONF.PATH.MODELS = os.path.join(CONF.PATH.BASE, "models")
CONF.PATH.UTILS = os.path.join(CONF.PATH.BASE, "utils")

# append to syspath
for _, path in CONF.PATH.items():
    sys.path.append(path)

# R3SCAN data
CONF.PATH.R3SCAN_META = os.path.join(CONF.PATH.R3SCAN, "meta_data")
CONF.PATH.R3SCAN_SCANS = '/data/3rscan/'
CONF.PATH.R3SCAN10dim_align = '/data/10dimPoints_align/'
CONF.PATH.R3SCAN10dim = '/data/10dimPoints'
CONF.PATH.H5 = "/221019046/Data/3RSCAN/processed/3rscan.pkl"
CONF.PATH.HF = "/221019046/Data/huggingface"

# CLEVR3D
CONF.PATH.CLEVR3D = os.path.join(CONF.PATH.BASE, 'data', "CLEVR3D")
CONF.PATH.CLEVER3D_data = os.path.join(CONF.PATH.CLEVR3D, "CLEVR3D-REAL.json")
CONF.PATH.CLEVER3D_answer = os.path.join(CONF.PATH.CLEVR3D, "answer_dict_x1.json")

# R3SCAN
CONF.PATH.R3SCAN_TRAIN = os.path.join(CONF.PATH.R3SCAN_META, "train.txt")
CONF.PATH.R3SCAN_VAL = os.path.join(CONF.PATH.R3SCAN_META, "val.txt")
