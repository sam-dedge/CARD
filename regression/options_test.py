import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import json
import torch
import numpy as np
import torch.utils.tensorboard as tb
import matplotlib.pyplot as plt
import time
from datetime import datetime, timezone, timedelta
from utils import *

torch.set_printoptions(sci_mode=False)

parser = argparse.ArgumentParser(description=globals()["__doc__"])

parser.add_argument(
    "--config", type=str, required=True, help="Path to the config file"
)
parser.add_argument('--device', type=int, default=0, help='GPU device id')
parser.add_argument('--thread', type=int, default=4, help='number of threads')
parser.add_argument("--seed", type=int, default=1234, help="Random seed")
parser.add_argument(
    "--exp", type=str, default="exp", help="Path for saving running related data."
)
parser.add_argument(
    "--doc",
    type=str,
    required=True,
    help="A string for documentation purpose. "
         "Will be the name of the log folder.",
)
parser.add_argument(
    "--comment", type=str, default="", help="A string for experiment comment"
)
parser.add_argument(
    "--verbose",
    type=str,
    default="info",
    help="Verbose level: info | debug | warning | critical",
)
parser.add_argument("--test", action="store_true", help="Whether to test the model")
parser.add_argument(
    "--sample",
    action="store_true",
    help="Whether to produce samples from the model",
)
parser.add_argument(
    "--train_guidance_only",
    action="store_true",
    help="Whether to only pre-train the guidance model f_phi",
)
parser.add_argument(
    "--run_all",
    action="store_true",
    help="Whether to run all train test splits",
)
parser.add_argument(
    "--noise_prior",
    action="store_true",
    help="Whether to apply a noise prior distribution at timestep T",
)
parser.add_argument(
    "--no_cat_f_phi",
    action="store_true",
    help="Whether to not concatenate f_phi as part of eps_theta input",
)
parser.add_argument("--fid", action="store_true")
parser.add_argument("--interpolation", action="store_true")
parser.add_argument(
    "--resume_training", action="store_true", help="Whether to resume training"
)
parser.add_argument(
    "-i",
    "--image_folder",
    type=str,
    default="images",
    help="The folder name of samples",
)
parser.add_argument(
    "--n_splits", type=int, default=20, help="total number of splits for a specific regression task"
)
parser.add_argument(
    "--split", type=int, default=0, help="which split to use for regression data"
)
parser.add_argument(
    "--init_split", type=int, default=0, help="initial split to train for regression data, usually for resume training"
)
parser.add_argument(
    "--rmse_timestep", type=int, default=0, help="selected timestep to report metric y RMSE"
)
parser.add_argument(
    "--qice_timestep", type=int, default=0, help="selected timestep to report metric y QICE"
)
parser.add_argument(
    "--picp_timestep", type=int, default=0, help="selected timestep to report metric y PICP"
)
parser.add_argument(
    "--nll_timestep", type=int, default=0, help="selected timestep to report metric y NLL"
)
parser.add_argument(
    "--ni",
    action="store_true",
    help="No interaction. Suitable for Slurm Job launcher",
)
parser.add_argument("--use_pretrained", action="store_true")
parser.add_argument(
    "--sample_type",
    type=str,
    default="generalized",
    help="sampling approach (generalized or ddpm_noisy)",
)
parser.add_argument(
    "--skip_type",
    type=str,
    default="uniform",
    help="skip according to (uniform or quadratic)",
)
parser.add_argument(
    "--timesteps", type=int, default=None, help="number of steps involved"
)
parser.add_argument(
    "--eta",
    type=float,
    default=0.0,
    help="eta used to control the variances of sigma",
)
parser.add_argument("--sequence", action="store_true")
# loss option
parser.add_argument(
    "--loss", type=str, default='ddpm', help="Which loss to use"
)
parser.add_argument("--nll_global_var", action="store_true",
                    help="Apply global variance for NLL computation")
parser.add_argument("--nll_test_var", action="store_true",
                    help="Apply sample variance of the test set for NLL computation")
# Conditional transport options
parser.add_argument(
    "--use_d",
    action="store_true",
    help="Whether to take an adversarially trained feature encoder",
)
parser.add_argument(
    "--full_joint",
    action="store_true",
    help="Whether to take fully joint matching",
)
parser.add_argument(
    "--num_sample", type=int, default=1, help="number of samples used in forward and reverse"
)
# Options Test Flag
parser.add_argument(
    "--opt_test", action="store_true", help="Options Test File Flag"
)
args = parser.parse_args()


def parse_config():
    # set log path
    args.log_path = os.path.join(args.exp, "logs", args.doc)
    # parse config file
    #* myconfig variable not used anywhere
    #myconfig = args.config + args.doc + '/config.yml'
    #print(myconfig, '>>>>>>>>>')
    with open(args.config + args.doc + '/config.yml', "r") as f:
        if args.sample or args.test or args.opt_test:
            config = yaml.unsafe_load(f)
            new_config = config
        else:
            config = yaml.safe_load(f)
            new_config = dict2namespace(config)

    tb_path = os.path.join(args.exp, "tensorboard", args.doc)

    if not args.test and not args.sample and not args.opt_test:
        args.im_path = os.path.join(args.exp, new_config.training.image_folder, args.doc)
        new_config.diffusion.noise_prior = True if args.noise_prior else False
        new_config.model.cat_y_pred = False if args.no_cat_f_phi else True
        if not args.resume_training:
            if not args.timesteps is None:
                new_config.diffusion.timesteps = args.timesteps
            if args.num_sample > 1:
                new_config.diffusion.num_sample = args.num_sample
            if os.path.exists(args.log_path):
                overwrite = False
                if args.ni:
                    overwrite = True
                else:
                    response = input("Folder {} already exists. Overwrite? (Y/N)".format(args.log_path))
                    if response.upper() == "Y":
                        overwrite = True

                if overwrite:
                    shutil.rmtree(args.log_path)
                    shutil.rmtree(tb_path)
                    shutil.rmtree(args.im_path)
                    os.makedirs(args.log_path)
                    os.makedirs(args.im_path)
                    if os.path.exists(tb_path):
                        shutil.rmtree(tb_path)
                else:
                    print("Folder exists. Program halted.")
                    sys.exit(0)
            else:
                os.makedirs(args.log_path)
                if not os.path.exists(args.im_path):
                    os.makedirs(args.im_path)

            with open(os.path.join(args.log_path, "config.yml"), "w") as f:
                yaml.dump(new_config, f, default_flow_style=False)

        new_config.tb_logger = tb.SummaryWriter(log_dir=tb_path)
        # setup logger
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError("level {} not supported".format(args.verbose))

        handler1 = logging.StreamHandler()
        handler2 = logging.FileHandler(os.path.join(args.log_path, "stdout.txt"))
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        handler1.setFormatter(formatter)
        handler2.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.addHandler(handler2)
        logger.setLevel(level)

    else:
        if args.sample:
            args.im_path = os.path.join(args.exp, new_config.sampling.image_folder, args.doc)
        elif args.opt_test:
            args.im_path = os.path.join(args.exp, new_config.testing.image_folder, args.doc)
        else:
            args.im_path = os.path.join(args.exp, new_config.testing.image_folder, args.doc)
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError("level {} not supported".format(args.verbose))

        handler1 = logging.StreamHandler()
        # saving test metrics to a .txt file
        handler2 = logging.FileHandler(os.path.join(args.log_path, "testmetrics.txt"))
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        handler1.setFormatter(formatter)
        handler2.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.addHandler(handler2)
        logger.setLevel(level)

        if args.sample or args.test:
            os.makedirs(args.im_path, exist_ok=True)

    # add device
    #device_name = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device_name = f'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set number of threads
    if args.thread > 0:
        torch.set_num_threads(args.thread)
        print('Using {} threads'.format(args.thread))

    # set random seed
    if args.run_all:
        if new_config.data.dataset != "uci":
            args.seed += 1  # apply a different seed for each run of toy example
    set_random_seed(args.seed)

    print('In main.py parse_config')
    print(new_config.data.dataset, '|', new_config.data)

    torch.backends.cudnn.benchmark = True

    return new_config, logger


def main():
    config, logger = parse_config()

    logging.info("Writing log file to {}".format(args.log_path))
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(args.comment))

    if args.loss == 'card_conditional':
        from card_regression import Diffusion
    else:
        raise NotImplementedError("Invalid loss option")

    try:
        runner = Diffusion(args, config, device=config.device)
        start_time = time.time()
        procedure = None
        if args.sample:
            runner.sample()
            procedure = "Sampling"
        elif args.test:
            print('----------------x--------------')
            y_rmse_all_steps_list, y_qice_all_steps_list, y_picp_all_steps_list, y_nll_all_steps_list = runner.test()
            procedure = "Testing"
        elif args.opt_test:
            print('----------------x--------------x----------------')
            print(type(runner), runner)
            procedure = "Options Testing"
        else:
            runner.train()
            procedure = "Training"
        end_time = time.time()
        logging.info("\n{} procedure finished. It took {:.4f} minutes.\n\n\n".format(
            procedure, (end_time - start_time) / 60))
        # remove logging handlers
        handlers = logger.handlers[:]
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()
        # return test metric lists
        if args.test:
            return 'hudibabab', y_rmse_all_steps_list, y_qice_all_steps_list, y_picp_all_steps_list, y_nll_all_steps_list, config
    except Exception:
        logging.error(traceback.format_exc())


if __name__ == "__main__":
    args.doc = args.doc + "/split_" + str(args.split)
    if args.opt_test:
        args.config = args.config + args.doc + "/config.yml"
    sys.exit(main())
