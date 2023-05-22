import os
import json
import torch
import datetime
import importlib
import numpy as np
import pytorch_lightning as pl

from lib.config import CONF
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from dataset import R3ScanVQAO27R16Dataset
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.profiler import SimpleProfiler
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


def get_CLEVER3D(num_scenes):
    # get initial scene list
    CLEVER3D = json.load(open(CONF.PATH.CLEVER3D_data))['questions']
    train_scene_list = np.loadtxt(CONF.PATH.R3SCAN_TRAIN, dtype=str).tolist()
    val_scene_list = np.loadtxt(CONF.PATH.R3SCAN_VAL, dtype=str).tolist()

    if num_scenes == -1:
        num_scenes = len(train_scene_list)
    else:
        assert len(train_scene_list) >= num_scenes

    # slice scene_list
    train_scene_list = train_scene_list[:num_scenes]
    val_scene_list = val_scene_list[:num_scenes]

    return CLEVER3D, train_scene_list, val_scene_list


def get_dataloader(args, CLEVER3D, scene_list, split, augment):
    dataset = R3ScanVQAO27R16Dataset(
        clever3d=CLEVER3D,
        r3scan_scene=scene_list[split],
        split=split,
        num_points=args.num_points,
        use_height=(not args.no_height),
        use_color=(not args.no_color),
        use_normal=args.use_normal,
        use_scene_graph=args.use_scene_graph,
        no_vision=args.no_vision,
        use_2d=args.use_2d,
        augment=augment,
        max_instance_num=args.max_instance_num,
        max_sentence_len=args.max_sentence_len,
        preloading=args.preloading,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=split == 'train',
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=dataset.collate_fn
    )

    return dataset, dataloader


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gpu', type=int, nargs='+', default=(0,), help='specify gpu devices')
    parser.add_argument("--seed", default=0, type=int)
    # data
    parser.add_argument('--num_scenes', type=int, default=-1, help='Number of scesne')
    parser.add_argument('--num_points', type=int, default=2048, help='Number of point in each instance')
    parser.add_argument('--max_instance_num', type=int, default=35, help='Max instance number in each scene')
    parser.add_argument('--max_sentence_len', type=int, default=32, help='Max question lenghth')
    # models
    parser.add_argument('--model', type=str, default='Transformer', help='specify model')
    parser.add_argument('--word_dropout', type=float, default=0.1, help="Dropout rate in word embedding.")
    parser.add_argument("--no_height", action="store_true", help="Do NOT use height signal in input.")
    parser.add_argument("--no_augment", action="store_true", help="Do NOT use data augmentation in input.")
    parser.add_argument("--no_color", action="store_true", help="Do NOT use RGB color in input.")
    parser.add_argument("--use_normal", action="store_true", help="Use normals in input.")
    parser.add_argument("--use_gt_cls", action="store_true", help="Use ground truth class as input.")
    parser.add_argument("--use_answer_type", action="store_true", help="Use ground truth answer class as input.")
    parser.add_argument("--use_scene_graph", action="store_true", help="Use scene graph as multi-task.")
    parser.add_argument("--use_gt_shape", action="store_true", help="Use ground truth shape as input.")
    parser.add_argument("--use_gt_color", action="store_true", help="Use ground truth color as input.")
    parser.add_argument("--use_gt_size", action="store_true", help="Use ground truth size as input.")
    parser.add_argument("--use_gt_material", action="store_true", help="Use ground truth material as input.")
    parser.add_argument("--use_2d", action="store_true", help="Use image sequences as input.")
    parser.add_argument("--no_vision", action="store_true", help="Do NOT use vision as input.")
    parser.add_argument("--preloading", action="store_true", help="Preloading dataset.")
    # training
    parser.add_argument('--log_dir', type=str, default='default', help='log location')
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="number of workers")
    parser.add_argument("--epoch", type=int, default=100, help="number of epochs")
    parser.add_argument('--init_lr', type=float, default=0.0001, help='learning rate for training.')
    parser.add_argument('--monitor_metric', type=str, default='val/ref_acc', help='metric to monitor')
    parser.add_argument('--stop_patience', type=int, default=10, help='Patience for stop training')
    parser.add_argument('--save_top_k', type=int, default=3, help='save top k checkpoints, use -1 to checkpoint every epoch')
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1, help='check_val_every_n_epoch')
    # testing
    parser.add_argument('--test', action='store_true', default=False, help='test mode')
    parser.add_argument('--checkpoint', type=str, default=None, help='load checkpoint')
    # debug
    parser.add_argument('--debug', default=False, action='store_true')
    args = parser.parse_args()
    print(args)

    # setting
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args.gpu))
    # os.environ['OMP_NUM_THREADS'] = str(args.num_workers)
    # os.environ['OPENBLAS_NUM_THREADS'] = str(args.num_workers)
    # os.environ['MKL_NUM_THREADS'] = str(args.num_workers)
    # os.environ['VECLIB_MAXIMUM_THREADS'] = str(args.num_workers)
    # os.environ['NUMEXPR_NUM_THREADS'] = str(args.num_workers)
    # torch.set_num_threads(args.num_workers)
    num_gpu = len(args.gpu)
    # output path
    tb_logger = pl_loggers.TensorBoardLogger('logs/', name=args.log_dir, default_hp_metric=False)
    os.makedirs(f'logs/{args.log_dir}', exist_ok=True)
    profiler = SimpleProfiler(output_filename=f'logs/{args.log_dir}/profiler.txt')
    np.set_printoptions(precision=4, suppress=True)

    # save the backup files
    backup_dir = os.path.join('logs', args.log_dir, 'backup_files_%s' % str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
    os.makedirs(backup_dir, exist_ok=True)
    os.system('cp main.py {}'.format(backup_dir))
    os.system('cp dataset/R3ScanDataset.py  {}'.format(backup_dir))
    os.system('cp models/{}.py {}'.format(args.model, backup_dir))

    # reproducibility
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(args.seed)

    args.num_scenes = 5 if args.debug else args.num_scenes
    CLEVER3D, scene_list_train, scene_list_val = get_CLEVER3D(args.num_scenes)
    all_scene_list = {
        "train": scene_list_train,
        "val": scene_list_val
    }


    # dataloader
    train_dataset, train_dataloader = get_dataloader(args, CLEVER3D, all_scene_list, "train", not args.no_augment)
    val_dataset, val_dataloader = get_dataloader(args, CLEVER3D, all_scene_list, "val", False)
    args.answer_classes = train_dataset.answer_classes_num

    dataloader = {
        "train": train_dataloader,
        "val": val_dataloader
    }

    # model
    model_file = importlib.import_module(args.model)  # import network module
    criterion = model_file.get_loss(args)
    net = model_file.get_model(args, criterion)

    if args.checkpoint is not None:
        print('load pre-trained model...')
        net.load_state_dict(torch.load(args.checkpoint)['state_dict'])

    pl.seed_everything(args.seed)
    checkpoint_callback = ModelCheckpoint(
        monitor='val/ref_acc',
        mode='max',
        save_last=True,
        save_top_k=args.save_top_k)

    if not args.test:
        # init trainer
        print('Start training...')
        trainer = pl.Trainer(gpus=[i for i in range(num_gpu)],
                             accelerator='ddp',
                             max_epochs=args.epoch,
                             resume_from_checkpoint=args.checkpoint,
                             callbacks=[checkpoint_callback,
                                        EarlyStopping(monitor=args.monitor_metric,
                                                      patience=args.stop_patience,
                                                      mode='max',
                                                      verbose=True)],
                             logger=tb_logger, profiler=profiler,
                             check_val_every_n_epoch=args.check_val_every_n_epoch,
                             gradient_clip_val=1)
        trainer.fit(net, train_dataloader, val_dataloader)

    else:
        print('Start testing...')
        trainer = pl.Trainer(gpus=[i for i in range(len(','.join(map(str, args.gpu))))], accelerator='ddp', logger=tb_logger, profiler=profiler, )
        trainer.validate(net, val_dataloader)
