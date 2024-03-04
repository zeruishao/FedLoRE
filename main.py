import os
import copy
import argparse
import torch

from torch import nn
from system.server.serverlore import FedLoRE
from system.trainmodel.models import BaseHeadSplit, CNN

torch.manual_seed(0)


# Training model
def load_model(args):
    args.model = CNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)


# Training algorithm
def train_model(args, i):
    args.head = copy.deepcopy(args.model.fc)
    args.model.fc = nn.Identity()
    args.model = BaseHeadSplit(args.model, args.head)
    server = FedLoRE(args, i)

    server.train()


# Training argparse
def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-algo', "--algorithm", type=str, default="FedLoRE")
    parser.add_argument('-data', "--dataset", type=str, default="Cifar100", choices=["Cifar10", "Cifar100"])
    parser.add_argument('-nb', "--num_classes", type=int, default=100, choices=["10", "100"])
    parser.add_argument('-nc', "--num_clients", type=int, default=100, help="Total number of clients")
    parser.add_argument('-m', "--model", type=str, default="cnn")
    parser.add_argument('-gr', "--global_rounds", type=int, default=150, help="Total number of communication rounds")
    parser.add_argument('-ls', "--local_epochs", type=int, default=5, help="Multiple update steps in one local epoch")
    parser.add_argument('-lbs', "--batch_size", type=int, default=256, help="Batch size for local training")
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.01, help="Local learning rate")
    parser.add_argument('-pv', "--prev", type=int, default=0, help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=5, help="Running times")
    parser.add_argument('-dev', "--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-prd', "--pre_round", type=int, default=5, help="Achieve the smooth training process")
    parser.add_argument('-rank', "--rank", type=int, default=5, help="Low-rank component estimation parameter")

    return parser.parse_args()


# Explain argparse
def exp_details(args):
    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    print("=" * 50)

    print("Algorithm: {}".format(args.algorithm))
    print("Dataset: {}".format(args.dataset))
    print("Number of classes: {}".format(args.num_classes))
    print("Total number of clients: {}".format(args.num_clients))
    print("Backbone: {}".format(args.model))
    print("Global rounds: {}".format(args.global_rounds))
    print("Local steps: {}".format(args.local_epochs))
    print("Local batch size: {}".format(args.batch_size))
    print("Local learing rate: {}".format(args.local_learning_rate))
    print("Running times: {}".format(args.times))
    print("Using device: {}".format(args.device))
    if args.device == "cuda":
        print("Cuda device id: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))

    print("=" * 50)


# Main training
def run(args):
    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")

        load_model(args)
        exp_details(args)
        train_model(args, i)

    print("\nAll done!")


if __name__ == "__main__":
    args = args_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    run(args)