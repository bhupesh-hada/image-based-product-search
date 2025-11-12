import argparse
import os
import os.path as osp
import random
import sys
import yaml

import torch
import numpy as np

import utils                                        # importing the functions present in the utils.py file

from tqdm import tqdm

from data_utils import get_dataloader               # from data_utils folder importing the get_dataloader class
from models import models                           # from models folder importing models file that has the load_model function 
from train import train, validation                 # from train.py importing the train and validation function
from utils import convert_dict_to_tuple             # from utils.py file importing the convert_dict_to_tuple


def main(args: argparse.Namespace) -> None:
    
    with open(args.cfg) as f:                        # reading the configuration file and converting it into a tuple
        data = yaml.safe_load(f)

    config = convert_dict_to_tuple(data)

    seed = config.dataset.seed                       # defining the seed value so that everytime the model runs, it gives the same output
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = False      # configuring the behavior of cudnn library 
    torch.backends.cudnn.benchmark = True

    outdir = osp.join(config.outdir, config.exp_name)        # creating an output directory for storing the model results after every epoch run 
    print("Savedir: {}".format(outdir))
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    train_loader, val_loader = get_dataloader.get_dataloaders(config)        # loading the training and validation data after proper augmentation 
                                                                             #  to introduce robustness in the training phase
    print("Loading model                                                     
    net = models.load_model(config)
    if config.num_gpu > 1:
        net = torch.nn.DataParallel(net)
    print("Done!")

    criterion, optimizer, scheduler = utils.get_training_parameters(config,  # getting the training parameters using the values specified in the configuration (cfg) file 
                                                                    net)
    train_epoch = tqdm(range(config.train.n_epoch), dynamic_ncols=True,      # to see the update in the model run after every 48 steps
                       desc='Epochs', position=0)

    # main process                                                           # getting the no. of epoch from the configuration file and lopping over each epoch to train the model and validate on the test dataset
    best_acc = 0.0
    for epoch in train_epoch:
        train(net, train_loader, criterion, optimizer, config, epoch)
        epoch_avg_acc = validation(net, val_loader, criterion, epoch)
        if epoch_avg_acc >= best_acc:                                        # keeping track of the best accuracy on the validation set 
            utils.save_checkpoint(net, optimizer, scheduler, epoch, outdir)  # after every epoch saving the model results in the .pth file so that it can be loaded to test its performance on the test_dev data (gallery and query images)
            best_acc = epoch_avg_acc
        scheduler.step()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='Path to config file.')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
