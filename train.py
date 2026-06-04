#!/usr/bin/env python3
"""
major actions here: fine-tune the features and evaluate different settings
"""
import os
import torch
import warnings

import numpy as np
import random

from time import sleep
from random import randint

import src.utils.logging as logging
from src.configs.config import get_cfg
from src.data import loader as data_loader
from src.engine.evaluator import Evaluator
from src.engine.trainer import Trainer
from src.models.build_model import build_model
from src.utils.file_io import PathManager

from diffusion import create_diffusion
from diffusers.models import AutoencoderKL

from launch import default_argument_parser, logging_train_setup
warnings.filterwarnings("ignore")

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # setup dist
    #cfg.DIST_INIT_PATH = "tcp://{}:12399".format(os.environ["SLURMD_NODENAME"])

    # setup output dir
    # output_dir / data_name / feature_name / lr_wd / run1
    output_dir = cfg.OUTPUT_DIR
    lr = cfg.SOLVER.BASE_LR
    wd = cfg.SOLVER.WEIGHT_DECAY
    prompt = cfg.MODEL.PROMPT.NUM_TOKENS 
    prompt_type = cfg.MODEL.PROMPT.TYPE
    mlp = cfg.MODEL.MLP_NUM
    t_list = "_" + "_".join(map(str, cfg.MODEL.T_LIST))
    layer_list = "_" + "_".join(map(str, cfg.MODEL.FEATURES_LAYER_LIST))
    feat_list = "_" + "_".join(map(str, cfg.MODEL.FEATURES_TYPE_LIST))
    if cfg.MODEL.FUSION_TYPE == "attention":
        fusion = int(cfg.MODEL.FUSION_ARC.split(',')[0].strip().split(':')[2])
        output_folder = os.path.join(
            cfg.DATA.NAME, cfg.DATA.FEATURE, f"t{t_list}_l{layer_list}_type{feat_list}_fusion_{fusion}_lr{lr}_wd{wd}_prompt{prompt_type}{prompt}_mlp{mlp}")
    elif cfg.MODEL.FUSION_TYPE == "linear":
        output_folder = os.path.join(
            cfg.DATA.NAME, cfg.DATA.FEATURE, f"t{t_list}_l{layer_list}_type{feat_list}_linear__lr{lr}_wd{wd}_prompt{prompt_type}{prompt}_mlp{mlp}")
    elif cfg.MODEL.FUSION_TYPE == "conv":
        output_folder = os.path.join(
            cfg.DATA.NAME, cfg.DATA.FEATURE, f"t{t_list}_l{layer_list}_type{feat_list}_conv__lr{lr}_wd{wd}_prompt{prompt_type}{prompt}_mlp{mlp}")

    # train cfg.RUN_N_TIMES times
    count = 1
    while count <= cfg.RUN_N_TIMES:
        output_path = os.path.join(output_dir, output_folder, f"run{count}")
        # pause for a random time, so concurrent process with same setting won't interfere with each other. # noqa
        #sleep(randint(3, 30))
        if not PathManager.exists(output_path):
            PathManager.mkdirs(output_path)
            cfg.OUTPUT_DIR = output_path
            break
        else:
            count += 1
    # if count > cfg.RUN_N_TIMES:
    #     raise ValueError(
    #         f"Already run {cfg.RUN_N_TIMES} times for {output_folder}, no need to run more")

    cfg.freeze()
    return cfg


def get_loaders(cfg, logger):
    logger.info("Loading training data (final training data for vtab)...")
    if cfg.DATA.NAME.startswith("vtab-"):
        #train_loader = data_loader.construct_trainval_loader(cfg)
        train_loader = data_loader.construct_train_loader(cfg)
    else:
        train_loader = data_loader.construct_train_loader(cfg)

    logger.info("Loading validation data...")
    # not really needed for vtab
    val_loader = data_loader.construct_val_loader(cfg)
    logger.info("Loading test data...")
    if cfg.DATA.NO_TEST:
        logger.info("...no test data is constructed")
        test_loader = None
    else:
        test_loader = data_loader.construct_test_loader(cfg)
    return train_loader,  val_loader, test_loader


def get_cifar100_dataloader(cfg, logger):
    # CIFAR
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    train_dataset = torchvision.datasets.CIFAR100(root='cfg.DATA.DATAPATH', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR100(root='cfg.DATA.DATAPATH', train=False, download=True, transform=transform)
    val_size = 0.1
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(val_size * num_train))

    # Shuffle indices
    np.random.shuffle(indices)

    train_idx, val_idx = indices[split:], indices[:split][:20]

    # Create samplers
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    train_loader = DataLoader(train_dataset, batch_size=cfg.DATA.BATCH_SIZE, sampler=train_sampler)
    val_loader = DataLoader(train_dataset, batch_size=cfg.DATA.BATCH_SIZE, sampler=val_sampler)
    test_loader = DataLoader(test_dataset, batch_size=cfg.DATA.BATCH_SIZE)
    logger.info(f"training size: {len(train_idx)}, validation size: {len(val_loader)}, test size: {len(test_dataset)}")

    return train_loader, val_loader, test_loader


def train(cfg, args):
    # clear up residual cache from previous runs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # main training / eval actions here

    # fix the seed for reproducibility
    if cfg.SEED is not None:
        torch.manual_seed(cfg.SEED)
        np.random.seed(cfg.SEED)
        random.seed(0)

    # setup training env including loggers
    logging_train_setup(args, cfg)
    logger = logging.get_logger("visual_prompt")

    train_loader, val_loader, test_loader = get_loaders(cfg, logger)
    #train_loader, val_loader, test_loader = get_cifar100_dataloader(cfg, logger)
    #test_loader = None
    #train_loader = test_loader
    logger.info("Constructing models...")
    model, cur_device = build_model(cfg)
    #model = model.half()  #make all weights in half precision


    logger.info("Setting up Evalutator...")
    evaluator = Evaluator()
    logger.info("Setting up Trainer...")
    trainer = Trainer(cfg, model, evaluator, cur_device)

    if train_loader:
        trainer.train_classifier(train_loader, val_loader, test_loader)
    else:
        print("No train loader presented. Exit")

    if cfg.SOLVER.TOTAL_EPOCH == 0:
        trainer.eval_classifier(test_loader, "test", 0)


def main(args):
    """main function to call from workflow"""

    # set up cfg and args
    cfg = setup(args)

    # Perform training.
    train(cfg, args)


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    main(args)
