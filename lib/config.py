import os
import sys
from easydict import EasyDict

CONF = EasyDict()

# path
CONF.PATH = EasyDict()
CONF.PATH.BASE = os.getcwd()
CONF.PATH.DATA = os.path.join(CONF.PATH.BASE, "data")
CONF.PATH.R3SCAN = os.path.join(CONF.PATH.DATA, "R3Scan")
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

# CLEVER3D
# CONF.PATH.CLEVER3D = os.path.join(CONF.PATH.R3SCAN, "CLEVER3D")
CONF.PATH.CLEVER3D = os.path.join(CONF.PATH.DATA, "CLEVR3D")
CONF.PATH.CLEVER3D_data = os.path.join(CONF.PATH.CLEVER3D, "CLEVR3D-REAL.json")

# R3SCAN
CONF.PATH.R3SCAN_TRAIN = os.path.join(CONF.PATH.R3SCAN_META, "train_scans.txt")
CONF.PATH.R3SCAN_VAL = os.path.join(CONF.PATH.R3SCAN_META, "validation_scans.txt")
