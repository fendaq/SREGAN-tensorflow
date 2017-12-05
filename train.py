#!/usr/bin/python
# -*- coding:utf-8 -*-

import data
import argparse
from model import EDSR


def loadArgu():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="/home/tengfei/Downloads/datasets/General-100")
    parser.add_argument("--imgsize", default=100, type=int)
    parser.add_argument("--scale", default=2, type=int)
    parser.add_argument("--layers", default=15, type=int)
    parser.add_argument("--featuresize", default=128, type=int)
    parser.add_argument("--batchsize", default=6, type=int)
    parser.add_argument("--savedir", default='saved_models')
    parser.add_argument("--iterations", default=100, type=int)
    args = parser.parse_args()

    return args


def main():
    args = loadArgu()
    data.load_dataset(args.dataset)  # get two list of train images and test images
    down_size = args.imgsize // args.scale
    network = EDSR(down_size, args.layers, args.featuresize, args.scale)
    network.set_data_fn(data.get_batch, (args.batchsize, args.imgsize, down_size), data.get_test_set,
                        (args.imgsize, down_size))
    network.train(args.iterations, args.savedir)

    return 1


if __name__ == '__main__':
    main()
