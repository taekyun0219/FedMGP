import argparse
import torch
import numpy as np
from Dassl.dassl.utils import setup_logger, set_random_seed, collect_env_info
from Dassl.dassl.config import get_cfg_default
import os
import gc
from federated_core.factory import FederatedLearnerFactory

def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.num_shots:
        cfg.DATASET.NUM_SHOTS = args.num_shots

    if args.subsample:
        cfg.DATASET.SUBSAMPLE_CLASSES = args.subsample
        
    if args.debug_mode is not None:
        cfg.DEBUG_MODE = args.debug_mode


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN
    
    cfg.DEBUG_MODE = False
    
    # Config for PromptFL
    cfg.TRAINER.PROMPTFL = CN()
    cfg.TRAINER.PROMPTFL.N_CTX = 16  # number of context vectors
    cfg.TRAINER.PROMPTFL.CTX_INIT = ""
    cfg.TRAINER.PROMPTFL.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.PROMPTFL.FEATURE = False
    cfg.TRAINER.PROMPTFL.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    # Config for CLIP (zero-shot inference)
    cfg.TRAINER.CLIP = CN()
    cfg.TRAINER.CLIP.PREC = "fp16"  # fp16, fp32, amp

    # Config for FEDPGP
    cfg.TRAINER.FEDPGP = CN()
    cfg.TRAINER.FEDPGP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.FEDPGP.CSC = False  # class-specific context
    cfg.TRAINER.FEDPGP.CTX_INIT = ""
    cfg.TRAINER.FEDPGP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.FEDPGP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
    cfg.TRAINER.FEDPGP.BOTTLENECK = 8
    cfg.TRAINER.FEDPGP.N = 2 # number of prompts
    cfg.TRAINER.FEDPGP.FEATURE = False
    cfg.TRAINER.FEDPGP.mu = 0.5
    cfg.TRAINER.FEDPGP.temp = 0.5

    # Config for FedMGP
    cfg.TRAINER.FEDMGP = CN()
    cfg.TRAINER.FEDMGP.N_CTX_VISION = 8
    cfg.TRAINER.FEDMGP.N_CTX_TEXT = 8
    cfg.TRAINER.FEDMGP.NUM_PROMPTS_VISION = 3
    cfg.TRAINER.FEDMGP.NUM_PROMPTS_TEXT = 3
    cfg.TRAINER.FEDMGP.CTX_INIT = ""
    cfg.TRAINER.FEDMGP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.FEDMGP.PROMPT_DEPTH_VISION = 1  # 1 for shallow prompting
    cfg.TRAINER.FEDMGP.PROMPT_DEPTH_TEXT = 1   # 1 for shallow prompting
    cfg.TRAINER.FEDMGP.USE_DIVERGENT_LOSS = False
    cfg.TRAINER.FEDMGP.DIVERGENT_LOSS_WEIGHT = 0.1
    cfg.TRAINER.FEDMGP.DIVERGENT_LOSS_TYPE = "l2"  # cos, l1, l2
    cfg.TRAINER.FEDMGP.TOPK = 2  # top-k prompts for aggregation
    cfg.TRAINER.FEDMGP.AGGREGATE_HIGHEST_SIM = False  # True: highest similarity, False: lowest
    cfg.TRAINER.FEDMGP.SELECT_MODE = "similarity"  # similarity, random, top_fixed, all
    cfg.TRAINER.FEDMGP.INFERENCE_MODE = "average"  # average, selected_group, max_logits, feature_average
    cfg.TRAINER.FEDMGP.SELECTED_PROMPT_GROUP = 0  # only used when INFERENCE_MODE=selected_group
    cfg.TRAINER.FEDMGP.RANDOM_SEED = 42
    cfg.TRAINER.FEDMGP.PERTURBATION_ENABLED = False  # add perturbation if same prompt selected 3 rounds
    cfg.TRAINER.FEDMGP.PROBABILISTIC_SELECTION = False  # probability sampling based on similarity
    cfg.TRAINER.FEDMGP.TEMPERATURE = 1.0  # softmax temperature for probability distribution

    # Config for FedOPT
    cfg.TRAINER.FEDOPT = CN()
    cfg.TRAINER.FEDOPT.N_CTX = 16  # number of context vectors
    cfg.TRAINER.FEDOPT.CSC = False  # class-specific context
    cfg.TRAINER.FEDOPT.CTX_INIT = ""  # initialization words
    cfg.TRAINER.FEDOPT.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.FEDOPT.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
    cfg.TRAINER.FEDOPT.N = 2 # number of prompts
    cfg.TRAINER.FEDOPT.THRESH = 0.001 # thresh of sinkhorn distance
    cfg.TRAINER.FEDOPT.EPS = 0.01 # lambada of sinkhorn distance
    cfg.TRAINER.FEDOPT.OT = "COT" # type of OT used
    cfg.TRAINER.FEDOPT.TOP_PERCENT = 0.80
    cfg.TRAINER.FEDOPT.MAX_ITER = 100

    # Config for PromptFolio
    cfg.TRAINER.PROMPTFOLIO = CN()
    cfg.TRAINER.PROMPTFOLIO.N_CTX = 16  # number of context vectors
    cfg.TRAINER.PROMPTFOLIO.CTX_INIT = ""  # initialization words
    cfg.TRAINER.PROMPTFOLIO.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.PROMPTFOLIO.N = 2 # number of prompts
    cfg.TRAINER.PROMPTFOLIO.FEATURE = False

    # Config for FedTPG
    cfg.TRAINER.FEDTPG = CN()
    cfg.TRAINER.FEDTPG.N_CTX = 4  # number of context vectors
    cfg.TRAINER.FEDTPG.D_CTX = 1
    cfg.TRAINER.FEDTPG.DEPTH = 0
    cfg.TRAINER.FEDTPG.PREC = "fp16"  # fp16, fp32, amp

    # Config for FedMoPG
    cfg.TRAINER.FEDMOPG = CN()
    cfg.TRAINER.FEDMOPG.N_CTX = 4  # number of context vectors in each prompt
    cfg.TRAINER.FEDMOPG.D_CTX = 1  # prompt insertion depth shared by text/vision
    cfg.TRAINER.FEDMOPG.DEPTH = 0  # self-attention depth inside generator
    cfg.TRAINER.FEDMOPG.NUM_PROMPT_PAIRS = 4  # G in mixture-of-prompt generation
    cfg.TRAINER.FEDMOPG.CROSS_HEADS = 4
    cfg.TRAINER.FEDMOPG.SELF_HEADS = 4
    cfg.TRAINER.FEDMOPG.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.FEDMOPG.USE_DIVERGENT_LOSS = False
    cfg.TRAINER.FEDMOPG.DIVERGENT_LOSS_WEIGHT = 0.1
    cfg.TRAINER.FEDMOPG.DIVERGENT_LOSS_TYPE = "cos"  # cos, l1, l2

    # Config for VPT
    cfg.TRAINER.VPT = CN()
    cfg.TRAINER.VPT.N_CTX_VISION = 4  # number of context vectors at the vision branch
    cfg.TRAINER.VPT.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.VPT.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.VPT.PROMPT_DEPTH_VISION = 1  # if set to 1, will represent shallow vision prompting only

    # Config for IVLP
    cfg.TRAINER.IVLP = CN()
    cfg.TRAINER.IVLP.N_CTX_VISION = 2  # number of context vectors at the vision branch
    cfg.TRAINER.IVLP.N_CTX_TEXT = 2  # number of context vectors at the language branch
    cfg.TRAINER.IVLP.CTX_INIT = "a photo of a"  # initialization words (only for language prompts)
    cfg.TRAINER.IVLP.PREC = "fp16"  # fp16, fp32, amp
    # If both variables below are set to 0, 0, will the config will degenerate to COOP model
    cfg.TRAINER.IVLP.PROMPT_DEPTH_VISION = 1 # Max 12, minimum 0, for 0 it will act as shallow MaPLe (J=1)
    cfg.TRAINER.IVLP.PROMPT_DEPTH_TEXT = 1  # Max 12, minimum 0, for 0 it will act as shallow MaPLe (J=1)
    cfg.TRAINER.IVLP.AGGREGATE = "all" # all, vision, language

    # Config for MaPLe
    cfg.TRAINER.MAPLE = CN()
    cfg.TRAINER.MAPLE.N_CTX = 2  # number of context vectors
    cfg.TRAINER.MAPLE.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.MAPLE.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.MAPLE.PROMPT_DEPTH = 9  # Max 12, minimum 0, for 1 it will act as shallow MaPLe (J=1)

    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COCOOP.PREC = "fp16"  # fp16, fp32, amp

    cfg.DATASET.SUBSAMPLE_CLASSES = "base"  # all, base or new
    cfg.DATASET.USERS = 10  # number of clients
    cfg.DATASET.IID = False  # is iid
    cfg.DATASET.FRAC = 1.0
    cfg.DATASET.PARTITION = "noniid-labeldir"
    cfg.DATASET.USEALL = False # use all data for training instead of few shot
    cfg.DATASET.NUM_SHOTS = 4
    cfg.DATASET.BETA = 0.3
    cfg.DATASET.REPEATRATE = 0.0 # repeat rate on each client
    cfg.DATALOADER.TRAIN_X.N_DOMAIN = 4 # number of domain
    cfg.DATASET.IMBALANCE_TRAIN = False # is adding label skew to feature skew datasets
    cfg.DATASET.SPLIT_CLIENT = False # is adding label skew to feature skew datasets and split one domain to multi clients
    cfg.OPTIM.ROUND = 10 # global round
    cfg.OPTIM.MAX_EPOCH = 2 # local epoch
    cfg.OPTIM.GAMMA = 1 # gamma of single-step
    cfg.OPTIM.LR = 0.001 #learning rate

    cfg.MODEL.BACKBONE.PRETRAINED = True


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)
    
    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)

    if cfg.SEED >= 0:
        set_random_seed(cfg.SEED)
        np.random.seed(cfg.SEED)
        torch.manual_seed(cfg.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cfg.SEED)
            torch.cuda.manual_seed_all(cfg.SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)

    federated_learner = FederatedLearnerFactory.create_learner(args.model, cfg, args)

    if args.eval_only or args.trainer == 'CLIP':
        if args.trainer == 'CLIP':
            print("CLIP trainer: zero-shot inference mode")

        if cfg.DATASET.SUBSAMPLE_CLASSES not in ['base', 'new', 'all']:
            print(f"Warning: SUBSAMPLE_CLASSES='{cfg.DATASET.SUBSAMPLE_CLASSES}', should be 'base', 'new', or 'all'")
            if args.subsample:
                print(f"Using subsample from args: '{args.subsample}'")
                cfg.DATASET.SUBSAMPLE_CLASSES = args.subsample
            else:
                print("Defaulting to 'new' for test classes")
                cfg.DATASET.SUBSAMPLE_CLASSES = 'new'

        print(f"Testing on {cfg.DATASET.SUBSAMPLE_CLASSES} classes")
        federated_learner.test()
    else:
        federated_learner.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # ================================== model+trainer determine the algorithm================================== 
    # fedavg: standard aggregation process, default model   local: no aggregation process
    parser.add_argument("--model", type=str, default="FedPGP", help="model of aggregation, choose from:fedavg, fedprox, local,FedPGP")
    # Option: CLIP, PromptFL, FedPGP
    parser.add_argument("--trainer", type=str, default="FedPGP", help="name of trainer, choose from: CLIP, PromptFL, FedPGP")

    # ================================== param for vision backbone ================================== 
    parser.add_argument("--config-file", type=str, default="configs/trainers/PLOT/vit_b16.yaml", help="path to config file")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")

    # ================================== param for dataset ================================== 
    parser.add_argument("--root", type=str, default="DATA/", help="path to dataset")
    parser.add_argument("--dataset-config-file", type=str, default="configs/datasets/oxford_pets.yaml", help="path to config file for dataset setup")

    # ================================== feaderal experimental settings ================================== 
    parser.add_argument("--seed", type=int, default=0, help="only positive value enables a fixed seed")
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument("--output-dir", type=str, default="outputtest/", help="output directory")
    parser.add_argument("--debug-mode", type=lambda x: (str(x).lower() == 'true'), default=None, help="Run test only in last round to speed up training")

    # ================================== data partition ================================== 
    parser.add_argument('--useall', default=False, help="is useall, True for all training samples, False for few shot learning")
    parser.add_argument('--num_shots', type=int, default=16, help="number of shots in few shot setting")
    parser.add_argument('--subsample', type=str, default='base', help="all,base,new")
    parser.add_argument("--transforms", type=str, nargs="+", help="data augmentation methods")

    # ================================== evaluation settings ================================== 
    parser.add_argument("--resume", type=str, default=None, help="checkpoint directory (from which the training resumes)")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument("--model-dir", type=str, default="", help="load model from this directory for eval-only mode")
    parser.add_argument("--load-epoch", type=int, help="load model weights at this epoch for evaluation")
    parser.add_argument("--no-train", action="store_true", help="do not call trainer.train()")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, help="modify config options using the command-line")

    
    args = parser.parse_args()
    main(args)






